#!/usr/bin/env python3
import argparse
import dataclasses
import json
import logging

import torch
from action_dict import GeneralizedActionDict
from data import (
    Dataset,
    SentencePieceVocabulary,
    SupervisedValResult,
)
from fixed_stack_models import (
    GeneralizedActionFixedStackRNNG,
    NoValidNextActionError,
    StackIndexError,
)
from utils import *

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()

# Data path options
parser.add_argument("--train_file", default="data/ptb-no-unk-train.json")
parser.add_argument("--val_file", default="data/ptb-no-unk-val.json")
parser.add_argument(
    "--sp_model",
    default="",
    help="Subword-tokenized treebank should be trained with this argument. Path to trained sentencepiece model.",
)
parser.add_argument("--save_path", default="rnng.pt", help="where to save the best model")
parser.add_argument(
    "--val_res_path",
    default="val_res.jsonl",
    help="Where to save the validation results.",
)

# Model options
parser.add_argument(
    "--strategy",
    help="Which strategy to use for training",
)
parser.add_argument("--stack_size", type=int, help="fixed stack size")
parser.add_argument(
    "--larger_stack_size",
    type=int,
    help="fixed stack size used when there is an error with the default stack size",
)
parser.add_argument("--w_dim", type=int, help="input/output word dimension")
parser.add_argument("--h_dim", type=int, help="LSTM hidden dimension")
parser.add_argument(
    "--num_layers",
    type=int,
    help="number of layers in LM and the stack LSTM (for RNNG)",
)
parser.add_argument("--dropout", type=float, help="dropout rate")
parser.add_argument(
    "--regularizer",
    choices=["layernorm", "dropout"],
    default="layernorm",
    help="Which regularization method to use for training",
)
parser.add_argument(
    "--composition",
    default="lstm",
    choices=["lstm", "attention"],
    help="lstm: original lstm composition; attention: gated attention introduced in Kuncoro et al. (2017).",
)

# Optimization options
parser.add_argument(
    "--batch_group",
    choices=["same_length", "random", "similar_length", "similar_action_length"],
    default="similar_length",
    help="Sentences are grouped by this criterion to make each batch.",
)
parser.add_argument(
    "--max_group_length_diff",
    default=20,
    type=int,
    help="When --batch_group=similar_length or similar_action_length, maximum (token or action) length difference in a single batch does not exceed this.",
)
parser.add_argument(
    "--group_sentence_size",
    default=1024,
    type=int,
    help="When --batch_group=similar_length, sentences are first sorted by length and grouped by this number of sentences, from which each batch is sampled.",
)
parser.add_argument(
    "--optimizer",
    default="adam",
    choices=["sgd", "adam"],
    help="Which optimizer to use.",
)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--val_batch_size", type=int, default=16)

parser.add_argument(
    "--batch_token_size",
    type=int,
    default=30000,
    help="Number of tokens in a batch (batch_size*sentence_length) does not exceed this.",
)
parser.add_argument(
    "--batch_action_size",
    type=int,
    default=100000,
    help="(batch_size*max_action_length) does not exceed this.",
)
parser.add_argument("--num_epochs", default=18, type=int, help="number of training epochs")
parser.add_argument("--num_steps", default=1000, type=int, help="number of training steps")
parser.add_argument("--lr", default=0.001, type=float, help="starting learning rate")
parser.add_argument(
    "--param_init",
    default=0,
    type=float,
    help="parameter initialization (over uniform)",
)
parser.add_argument("--max_grad_norm", default=5, type=float, help="gradient clipping parameter")
parser.add_argument("--gpu", default=0, type=int, help="which gpu to use")
parser.add_argument(
    "--device",
    default="cuda",
    choices=["cuda", "cpu"],
    help='If "cuda", GPU number --gpu is used.',
)
parser.add_argument("--seed", default=3435, type=int, help="random seed")
parser.add_argument("--print_every", type=int, default=500, help="print stats after this many batches")
parser.add_argument(
    "--valid_every",
    type=int,
    default=-1,
    help="If > 0, validate and save model every this many batches",
)


def to_namespace(args):
    if isinstance(args, dict):
        # Args is saved as a dict so we need to convert to Namespace when loading from a checkpoint.
        from argparse import Namespace

        args = Namespace(**args)
    return args


def create_optimizer(args, model):
    args = to_namespace(args)
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        # The following was in the original batched rnng implementation; but the necessity is not clear. So we remove it.
        # args.min_epochs = args.num_epochs
    return optimizer


def create_model(
    args,
    action_dict: GeneralizedActionDict,
    vocab: SentencePieceVocabulary,
):
    args = to_namespace(args)

    model = GeneralizedActionFixedStackRNNG(
        action_dict=action_dict,
        vocab_size=vocab.size(),
        vocab_padding_idx=vocab.padding_idx,
        w_dim=args.w_dim,
        h_dim=args.h_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        layernorm=args.regularizer == "layernorm",
        attention_composition=args.composition == "attention",
    )
    if args.param_init > 0:
        for param in model.parameters():
            param.data.uniform_(-args.param_init, args.param_init)
    return model


def validate_supervised(
    val_data: Dataset,
    model: GeneralizedActionFixedStackRNNG,
    epoch: int,
    global_step: int,
    args,
    # Args used for saving checkpoint.
    val_losses_supervised: list[float],
    optimizer: torch.optim.SGD | torch.optim.Adam,
    train_data: Dataset,
):
    device = next(model.parameters()).device
    model.eval()

    logger.info("--------------------------------")
    logger.info("Checking supervised validation performances...")

    num_sents: int = 0
    total_nt_pos_count: dict[int, int] = dict()

    # Here, we calculate the ppl per sentence.
    total_supervised_loss: float = 0.0
    total_supervised_token_loss: float = 0.0
    total_supervised_action_loss: float = 0.0
    total_supervised_token_ppl: float = 0.0
    total_supervised_action_ppl: float = 0.0

    with torch.no_grad():
        # Do not need to specify strategy for Dataset.batches to avoid loading strategy specific data.
        for batch in val_data.batches(
            batch_size=args.val_batch_size,
            batch_token_size=args.batch_token_size,
            batch_action_size=args.batch_action_size,
            batch_group=args.batch_group,
            max_length_diff=args.max_group_length_diff,
            group_sentence_size=args.group_sentence_size,
            shuffle=False,
        ):
            (
                token_ids,
                given_action_ids,
                batch_idx,
            ) = batch
            given_action_ids = given_action_ids.to(device=device)
            token_ids = token_ids.to(device=device)

            try_again: bool = False
            try:
                # Validation by supervised loss.
                (
                    supervised_loss,
                    supervised_token_loss,
                    supervised_action_loss,
                    supervised_token_ppl,
                    supervised_action_ppl,
                    nt_pos_count,
                ) = model.supervised_forward(
                    x=token_ids,
                    actions=given_action_ids,
                    stack_size=args.stack_size,  # Use default stack size first.
                )

            except (NoValidNextActionError, StackIndexError):
                logger.warning(
                    f"The problem may or may not be because the default stack size {args.stack_size} is not enough."
                )
                logger.warning(
                    f"Try using larger stack size {args.larger_stack_size} to avoid NoValidNextActionError in get_invalid_action_mask or StackIndexError in Stack."
                )
                logger.warning(
                    "If the problem does not solve by increasing stack size, the cause of the problem may not be stack size."
                )
                # A bit hacky, but rerun must be done out of except clause.
                # https://pytorch.org/docs/stable/notes/faq.html#my-out-of-memory-exception-handler-can-t-allocate-memory
                try_again = True

            if try_again:
                # Validation by supervised loss.
                (
                    supervised_loss,
                    supervised_token_loss,
                    supervised_action_loss,
                    supervised_token_ppl,
                    supervised_action_ppl,
                    nt_pos_count,
                ) = model.supervised_forward(
                    x=token_ids,
                    actions=given_action_ids,
                    stack_size=args.larger_stack_size,  # Use default stack size first.
                )

            # Add validation scores by inference.
            num_sents += token_ids.size(0)
            total_nt_pos_count = update_nt_pos_count(nt_pos_count=total_nt_pos_count, new_counts=nt_pos_count)

            # Add supervised validation loss.
            total_supervised_loss += supervised_loss.sum().detach().item()
            total_supervised_token_loss += supervised_token_loss.sum().detach().item()
            total_supervised_action_loss += supervised_action_loss.sum().detach().item()
            total_supervised_token_ppl += supervised_token_ppl.sum().detach().item()
            total_supervised_action_ppl += supervised_action_ppl.sum().detach().item()

    total_nt_count = sum([c for c in total_nt_pos_count.values()])

    eval_results = SupervisedValResult(
        epoch=epoch,
        step=global_step,
        nt_pos_count={pos: count / total_nt_count for pos, count in total_nt_pos_count.items()},
        supervised_loss=total_supervised_loss,
        supervised_token_loss=total_supervised_token_loss,
        supervised_action_loss=total_supervised_action_loss,
        supervised_token_ppl=(total_supervised_token_ppl / num_sents),
        supervised_action_ppl=(total_supervised_action_ppl / num_sents),
    )

    # Back to train mode.
    model.train()

    # Log validateion results.
    logger.info(
        "SupervisedLoss: {:.4f}, "
        "SupervisedActionLoss: {:.4f}, "
        "SupervisedTokenLoss: {:.4f}, "
        "SupervisedActionPPL: {:.2f}, "
        "SupervisedTokenPPL: {:.2f}, "
        "NTPos: {}".format(
            eval_results.supervised_loss,
            eval_results.supervised_action_loss,
            eval_results.supervised_token_loss,
            eval_results.supervised_action_ppl,
            eval_results.supervised_token_ppl,
            get_nt_pos_count_str(eval_results.nt_pos_count),
        )
    )
    logger.info("--------------------------------")

    # Save model if best validation performance is updated.
    best_val_loss_supervised = float("inf") if len(val_losses_supervised) == 0 else min(val_losses_supervised)
    cur_val_loss_supervised = eval_results.supervised_loss

    if cur_val_loss_supervised < best_val_loss_supervised:
        best_val_loss_supervised = cur_val_loss_supervised
        checkpoint = {
            "train_args": args.__dict__,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "vocab": train_data.vocab,
            "prepro_args": train_data.prepro_args,
            "action_dict": train_data.action_dict,
            "epoch": epoch,
        }
        logger.info("Saving checkpoint to {}".format(args.save_path))
        torch.save(checkpoint, args.save_path)

    val_losses_supervised.append(cur_val_loss_supervised)

    # Save the validation results.
    with open(args.val_res_path, mode="a") as f:
        dict_str: str = json.dumps(dataclasses.asdict(eval_results))
        f.write("\n" + dict_str)


def print_pretrain_loss(
    epoch: int,
    global_step: int,
    model: GeneralizedActionFixedStackRNNG,
    optimizer: torch.optim.SGD | torch.optim.Adam,
    total_loss: float,
    total_token_loss: float,
    total_action_loss: float,
    total_token_ppl: float,
    total_action_ppl: float,
    batches_count: int,
    num_batches: int | None,
    total_batch_count: int,
    total_nt_pos_count: dict[int, int],
):
    param_norm = sum([p.norm() ** 2 for p in model.parameters()]).item() ** 0.5

    total_nt_count = sum([c for c in total_nt_pos_count.values()])
    nt_pos_count: dict[int, float] = {pos: count / total_nt_count for pos, count in total_nt_pos_count.items()}

    logger.info(
        "Epoch: {}, Step: {}, Batch: {}/{}, LR: {:.4f}, "
        "Loss: {:.2f}, "
        "TokenLoss: {:.2f}, "
        "ActionLoss: {:.2f}, "
        "TokenPPL: {:.2f}, "
        "ActionPPL: {:.2f}, "
        "|Param|: {:.2f}, "
        "NTPos: {}".format(
            epoch,
            global_step,
            batches_count,
            num_batches,
            optimizer.param_groups[0]["lr"],
            total_loss / total_batch_count,
            total_token_loss / total_batch_count,
            total_action_loss / total_batch_count,
            total_token_ppl / total_batch_count,
            total_action_ppl / total_batch_count,
            param_norm,
            get_nt_pos_count_str(nt_pos_count=nt_pos_count),
        )
    )


def supervised_pretrain(
    train_data: Dataset,
    val_data: Dataset,
    model: GeneralizedActionFixedStackRNNG,
    optimizer: torch.optim.SGD | torch.optim.Adam,
    epoch: int,
    global_step: int,
    val_losses_supervised: list[float],
    device: str,
    args,
):
    logger.info("Start supervised pretrain!")

    while epoch <= args.num_epochs or global_step <= args.num_steps:
        batches_count: int = 0  # batch count for the epoch. This does not reset after printing the reults.

        # Take total and average for printing.
        total_loss: float = 0.0
        total_token_loss: float = 0.0
        total_action_loss: float = 0.0
        total_token_ppl: float = 0.0
        total_action_ppl: float = 0.0
        total_nt_pos_count: dict[int, int] = dict()
        total_batch_count: int = 0  # This value is reset after printing the results.

        for batch in train_data.batches(
            batch_size=args.batch_size,
            batch_token_size=args.batch_token_size,
            batch_action_size=args.batch_action_size,
            batch_group=args.batch_group,
            max_length_diff=args.max_group_length_diff,
            group_sentence_size=args.group_sentence_size,
            shuffle=True,
        ):
            # Update global step.
            global_step += 1

            # Quit training if the steps reaches num_steps the epochs reaches num_epochs.
            if global_step > args.num_steps and epoch > args.num_epochs:
                break

            optimizer.zero_grad()

            (
                token_ids,
                given_action_ids,
                batch_idx,
            ) = batch

            token_ids = token_ids.to(device=device)
            given_action_ids = given_action_ids.to(device=device)

            try_again: bool = False
            try:
                loss, token_loss, action_loss, token_ppl, action_ppl, nt_pos_count = model.supervised_forward(
                    x=token_ids,
                    actions=given_action_ids,
                    stack_size=args.stack_size,  # Use default stack size first.
                )
            except (NoValidNextActionError, StackIndexError):
                logger.warning(
                    f"The problem may or may not be because the default stack size {args.stack_size} is not enough."
                )
                logger.warning(
                    f"Try using larger stack size {args.larger_stack_size} to avoid NoValidNextActionError in get_invalid_action_mask or StackIndexError in Stack."
                )
                logger.warning(
                    "If the problem does not solve by increasing stack size, the cause of the problem may not be stack size."
                )
                # A bit hacky, but rerun must be done out of except clause.
                # https://pytorch.org/docs/stable/notes/faq.html#my-out-of-memory-exception-handler-can-t-allocate-memory
                try_again = True

            if try_again:
                loss, token_loss, action_loss, token_ppl, action_ppl, nt_pos_count = model.supervised_forward(
                    x=token_ids,
                    actions=given_action_ids,
                    stack_size=args.larger_stack_size,  # Use (given) larger stack size next.
                )

            # Backpropagate.
            loss_to_backprop = loss.mean()  # Mean over batches.
            loss_to_backprop.backward()

            # Update variables for printing.
            total_loss += loss_to_backprop.detach().item()
            total_token_loss += token_loss.mean().detach().item()  # Mean over batches.
            total_token_ppl += token_ppl.mean().detach().item()  # Mean over batches.
            total_action_ppl += action_ppl.mean().detach().item()  # Mean over batches.
            total_action_loss += action_loss.mean().detach().item()  # Mean over batches.
            total_nt_pos_count = update_nt_pos_count(nt_pos_count=total_nt_pos_count, new_counts=nt_pos_count)
            batches_count += 1
            total_batch_count += 1

            # Optimizer step.
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            # Print losses.
            if global_step % args.print_every == 0:
                print_pretrain_loss(
                    epoch=epoch,
                    global_step=global_step,
                    model=model,
                    optimizer=optimizer,
                    total_loss=total_loss,
                    total_token_loss=total_token_loss,
                    total_action_loss=total_action_loss,
                    total_token_ppl=total_token_ppl,
                    total_action_ppl=total_action_ppl,
                    batches_count=batches_count,
                    num_batches=train_data.num_batches,
                    total_nt_pos_count=total_nt_pos_count,
                    total_batch_count=total_batch_count,
                )
                # Reset variables.
                # batches_count: int = 0
                total_loss: float = 0.0
                total_token_loss: float = 0.0
                total_action_loss: float = 0.0
                total_token_ppl: float = 0.0
                total_action_ppl: float = 0.0
                total_nt_pos_count: dict[int, int] = dict()
                total_batch_count: int = 0

                print_grad(model=model)

            # Do validation.
            if args.valid_every > 0 and global_step % args.valid_every == 0:
                validate_supervised(
                    val_data=val_data,
                    model=model,
                    epoch=epoch,
                    global_step=global_step,
                    args=args,
                    val_losses_supervised=val_losses_supervised,
                    optimizer=optimizer,
                    train_data=train_data,
                )

        if global_step <= args.num_steps and global_step % args.print_every != 0:
            # Otherwise training is not performed, so no need to print the outputs.
            # The second term is just to prevent printing the same log twice.
            print_pretrain_loss(
                epoch=epoch,
                global_step=global_step,
                model=model,
                optimizer=optimizer,
                total_loss=total_loss,
                total_token_loss=total_token_loss,
                total_action_loss=total_action_loss,
                total_token_ppl=total_token_ppl,
                total_action_ppl=total_action_ppl,
                batches_count=batches_count,
                num_batches=train_data.num_batches,
                total_nt_pos_count=total_nt_pos_count,
                total_batch_count=total_batch_count,
            )

        # Do validation after each epoch (when valida_every <= 0).
        if args.valid_every <= 0:
            validate_supervised(
                val_data=val_data,
                model=model,
                epoch=epoch,
                global_step=global_step,
                args=args,
                val_losses_supervised=val_losses_supervised,
                optimizer=optimizer,
                train_data=train_data,
            )

        # Update epoch.
        epoch += 1

    # Last validation is necessary when validations were performed intermediately.
    if args.valid_every > 0 and global_step % args.valid_every != 0:
        validate_supervised(
            val_data=val_data,
            model=model,
            epoch=epoch,
            global_step=global_step,
            args=args,
            val_losses_supervised=val_losses_supervised,
            optimizer=optimizer,
            train_data=train_data,
        )

    logger.info("Finished supervised pretrain!")


def main(args):
    logger.info("Args: {}".format(args))
    # Set random seed.
    set_random_seed(random_seed=args.seed)

    # Set device.
    if args.device == "cuda":
        device = "cuda:{}".format(args.gpu)
    else:
        device = "cpu"

    if len(args.sp_model) > 0:
        logger.info("Load sentencepiece vocabulary from {}".format(args.sp_model))
        vocab = SentencePieceVocabulary(args.sp_model)
    else:
        vocab = None

    # Prepare datasets.
    train_data = Dataset.from_json(
        data_file=args.train_file,
        strategy=args.strategy,
        vocab=vocab,
        action_dict=None,  # Create new action_dict inside Dataset.
        load_all_info=False,
    )
    vocab = train_data.vocab
    action_dict = train_data.action_dict
    val_data = Dataset.from_json(
        data_file=args.val_file,
        strategy=args.strategy,
        vocab=vocab,
        action_dict=action_dict,
        load_all_info=False,
    )
    vocab_size = int(train_data.vocab_size)
    logger.info("Train: %d sents, Val: %d sents" % (len(train_data.sents), len(val_data.sents)))
    logger.info("Vocab size: %d" % vocab_size)

    epoch = 1
    global_step = 0
    val_losses_supervised: list[float] = []

    # Prepare model.
    model = create_model(args, action_dict, vocab).to(device)
    optimizer = create_optimizer(args, model)

    logger.info("model architecture")
    logger.info(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model total parameters: {}".format(total_params))
    model.train()

    # Prepare empty val_res file.
    # Truncate existing val_res file.
    with open(args.val_res_path, mode="w"):
        pass

    # First, validate before training.
    logger.info("Validate before training!!!!")
    validate_supervised(
        val_data=val_data,
        model=model,
        epoch=epoch,
        global_step=global_step,
        args=args,
        val_losses_supervised=val_losses_supervised,
        optimizer=optimizer,
        train_data=train_data,
    )

    # Supervised pretrain.
    supervised_pretrain(
        train_data=train_data,
        val_data=val_data,
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        global_step=global_step,
        val_losses_supervised=val_losses_supervised,
        device=device,
        args=args,
    )


if __name__ == "__main__":
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s:%(name)s:%(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler("{}.log".format(args.save_path)),
            logging.StreamHandler(),
        ],
    )

    main(args)
