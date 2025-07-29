import argparse
import json
import logging
import time

import torch
from action_dict import GeneralizedActionDict
from data import Dataset, SentencePieceVocabulary
from fixed_stack_models import (
    GeneralizedActionFixedStackRNNG,
    NoValidNextActionError,
    StackIndexError,
)
from tqdm import tqdm
from utils import *

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()

# Data path options
parser.add_argument("--test_file", default="data/ptb-test.raw.txt")
parser.add_argument("--output_file", default="beam_seach.jsonl")
parser.add_argument("--model_file", default="rnng.pt")

# Beam search options
parser.add_argument("--beam_size", type=int)
parser.add_argument(
    "--min_shift_size",
    type=int,
    help="minimum number of shift actions to force at each time step during word sync beam search",
)
parser.add_argument(
    "--word_sync_step_limit",
    type=int,
    help="Inner iteration limit for each word sync step",
)
parser.add_argument("--stack_size", type=int, help="fixed stack size")
parser.add_argument(
    "--larger_stack_size",
    type=int,
    help="fixed stack size used when there is an error with the default stack size",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=10,
    help="Please decrease this if memory error occurs.",
)
parser.add_argument("--block_size", type=int, default=100)
parser.add_argument(
    "--batch_token_size",
    type=int,
    default=300000000,  # No limit
    help="Number of tokens in a batch (batch_size*sentence_length) does not exceed this. This value could be large value (e.g., 10000) safely when --stack_size_bound is set to some value > 0. Otherwise, we need to control (decrease) the batch size for longer sentences using this option, because then stack_size_bound will grow by sentence length.",
)
parser.add_argument(
    "--batch_action_size",
    type=int,
    default=100000000,  # No limit
    help="(batch_size*max_action_length) does not exceed this.",
)
parser.add_argument(
    "--group_sentence_size",
    default=102400,  # No limit
    type=int,
    help="When --batch_group=similar_length, sentences are first sorted by length and grouped by this number of sentences, from which each batch is sampled.",
)
parser.add_argument("--gpu", default=0, type=int, help="which gpu to use")
parser.add_argument(
    "--device",
    default="cuda",
    choices=["cuda", "cpu"],
    help='If "cuda", GPU number --gpu is used.',
)
parser.add_argument("--seed", default=3435, type=int)
parser.add_argument("--fp16", action="store_true")
parser.add_argument(
    "--max_length_diff",
    default=2000000,  # No limit
    type=int,
    help="Maximum sentence length difference in a single batch does not exceed this.",
)


def load_model(
    checkpoint,
    action_dict: GeneralizedActionDict,
    vocab: SentencePieceVocabulary,
):
    if "model_state_dict" in checkpoint:
        from train import create_model

        model = create_model(checkpoint["train_args"], action_dict, vocab)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model
    else:
        return checkpoint["model"]


def main(args):
    if args.device == "cuda":
        device = "cuda:{}".format(args.gpu)
    else:
        device = "cpu"

    # Set random seed.
    set_random_seed(random_seed=args.seed)

    if args.device == "cuda":
        torch.cuda.manual_seed(args.seed)

    checkpoint = torch.load(args.model_file)
    vocab: SentencePieceVocabulary = checkpoint["vocab"]
    action_dict: GeneralizedActionDict = checkpoint["action_dict"]
    prepro_args = checkpoint["prepro_args"]
    model: GeneralizedActionFixedStackRNNG = load_model(checkpoint, action_dict, vocab).to(device)

    if args.fp16:
        model.half()

    dataset = Dataset.from_text_file(
        text_file=args.test_file,
        vocab=vocab,
        action_dict=action_dict,
        prepro_args=prepro_args,
    )
    logger.info("model architecture")
    logger.info(model)
    model.eval()

    cur_block_size: int = 0

    all_parses: list[str] = []
    all_best_actions: list[list[int]] = []

    all_beam_scores: list[list[list[float]]] = []
    all_beam_token_log_probs: list[list[list[float]]] = []

    def sort_and_print_trees(
        block_idxs: list[int],
        block_parses: list[str],
        block_best_actions: list[list[int]],
        block_beam_scores: list[list[list[float]]],
        block_beam_token_log_probs: list[list[list[float]]],
    ):
        parse_idx_to_sent_idx = sorted(list(enumerate(block_idxs)), key=lambda x: x[1])

        orig_order_parses = [block_parses[parse_idx] for (parse_idx, _) in parse_idx_to_sent_idx]

        orig_order_best_actions = [block_best_actions[parse_idx] for (parse_idx, _) in parse_idx_to_sent_idx]

        orig_order_beam_scores = [block_beam_scores[parse_idx] for (parse_idx, _) in parse_idx_to_sent_idx]
        orig_order_beam_token_log_probs = [
            block_beam_token_log_probs[parse_idx] for (parse_idx, _) in parse_idx_to_sent_idx
        ]

        all_parses.extend(orig_order_parses)
        all_best_actions.extend(orig_order_best_actions)

        all_beam_scores.extend(orig_order_beam_scores)
        all_beam_token_log_probs.extend(orig_order_beam_token_log_probs)

        for parse in orig_order_parses:
            print(parse)

    def print_best_actions(
        best_actions: list[list[int]],  # (batch_size, max_action_len)
    ):
        action_str_l: list[list[str]] = []
        for action_ids in best_actions:
            action_str: list[str] = [action_dict.i2a(action_id) for action_id in action_ids]
            action_str_l.append(action_str)

        print(f"best_actions={action_str_l}")

    # Inference.
    start_time = time.time()
    with torch.no_grad():
        block_idxs: list[int] = []
        block_parses: list[str] = []
        block_best_actions: list[list[int]] = []

        block_beam_scores: list[list[list[float]]] = []
        block_beam_token_log_probs: list[list[list[float]]] = []

        batches = [
            batch
            for batch in dataset.test_batches(
                batch_size=args.batch_size,
                batch_token_size=args.batch_token_size,
                batch_action_size=args.batch_action_size,
                block_size=args.block_size,
                group_sentence_size=args.group_sentence_size,
                max_length_diff=args.max_length_diff,
            )
        ]

        for batch in tqdm(batches):
            tokens, batch_idx = batch
            tokens = tokens.to(device)

            try_again: bool = False
            try:
                best_actions, beam_scores, beam_token_log_probs = model.inference(
                    x=tokens,
                    stack_size=args.stack_size,  # Use default stack size first.
                    beam_size=args.beam_size,  # For now, we force to use the same beam size for both open/step_complete_beam.
                    min_shift_size=args.min_shift_size,
                    word_sync_step_limit=args.word_sync_step_limit,
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
                best_actions, beam_scores, beam_token_log_probs = model.inference(
                    x=tokens,
                    stack_size=args.larger_stack_size,  # Use (given) larger stack size next.
                    beam_size=args.beam_size,  # For now, we force to use the same beam size for both open/step_complete_beam.
                    min_shift_size=args.min_shift_size,
                    word_sync_step_limit=args.word_sync_step_limit,
                )

            print()
            print_best_actions(best_actions=best_actions)
            print()

            trees = [
                action_dict.build_tree_str(
                    action_ids=best_actions[i],
                    tokens=dataset.sents[batch_idx[i]].tokens_w_eos,
                )
                for i in range(len(batch_idx))
            ]
            block_idxs.extend(batch_idx)
            block_parses.extend(trees)
            block_best_actions.extend(best_actions)

            block_beam_scores.extend(beam_scores)
            block_beam_token_log_probs.extend(beam_token_log_probs)

            cur_block_size += tokens.size(0)

            if cur_block_size >= args.block_size:
                assert cur_block_size == args.block_size
                sort_and_print_trees(
                    block_idxs=block_idxs,
                    block_parses=block_parses,
                    block_best_actions=block_best_actions,
                    block_beam_scores=block_beam_scores,
                    block_beam_token_log_probs=block_beam_token_log_probs,
                )
                block_idxs = []
                block_parses = []
                block_best_actions = []
                block_beam_scores = []
                block_beam_token_log_probs = []
                cur_block_size = 0

    sort_and_print_trees(
        block_idxs=block_idxs,
        block_parses=block_parses,
        block_best_actions=block_best_actions,
        block_beam_scores=block_beam_scores,
        block_beam_token_log_probs=block_beam_token_log_probs,
    )
    end_time = time.time()

    with open(args.output_file, mode="wt") as o:
        for i in range(len(all_best_actions)):
            sent_res = {}
            sent_res["best_actions"] = all_best_actions[i]

            sent_res["beam_scores"] = all_beam_scores[i]
            sent_res["beam_token_log_probs"] = all_beam_token_log_probs[i]

            assert len(all_beam_scores[i]) == len(dataset.sents[i].token_ids)
            assert len(all_beam_token_log_probs[i]) == len(dataset.sents[i].token_ids)

            # Write the results in jsonl style.
            o.write(json.dumps(sent_res) + "\n")

        print(
            "Time: {} Throughput: {}".format(
                end_time - start_time,
                (len(dataset.sents)) / (end_time - start_time),
            )
        )


if __name__ == "__main__":
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s:%(name)s:%(levelname)s: %(message)s",
    )

    main(args)
