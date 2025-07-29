import argparse
import dataclasses
import json
import logging
from dataclasses import dataclass

import numpy as np
import torch
import utils
from action_dict import GeneralizedActionDict
from data import EvalCorpusResults, EvalDataset, SentencePieceVocabulary
from fixed_stack_models import (
    GeneralizedActionFixedStackRNNG,
    NoValidNextActionError,
    StackIndexError,
)
from inference import load_model
from utils import *

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()

# Data path options
# inference_file, gold_tree_file, and text_file corresponds to each other line by line.
parser.add_argument("--inference_file", required=True)
parser.add_argument("--gold_tree_file", default="")
parser.add_argument("--text_file", default="")
parser.add_argument("--model_file", default="rnng.pt")
parser.add_argument("--output_file", default="results.json")
parser.add_argument("--strategy_params_file", default="")

# Evaluation options
parser.add_argument(
    "--strategy",
    help="The strategy used for beam search",
)
parser.add_argument("--stack_size", type=int, help="fixed stack size")
parser.add_argument(
    "--larger_stack_size",
    type=int,
    help="fixed stack size used when there is an error with the default stack size",
)

parser.add_argument("--filter_by_parsability", action="store_true")
parser.add_argument("--word_sync_step_limit", type=int)
parser.add_argument("--gpu", default=0, type=int, help="which gpu to use")
parser.add_argument(
    "--device",
    default="cuda",
    choices=["cuda", "cpu"],
    help='If "cuda", GPU number --gpu is used.',
)
parser.add_argument("--seed", default=3435, type=int)
parser.add_argument("--fp16", action="store_true")


def calc_parsing_scores(
    dataset: EvalDataset,
) -> tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
]:
    corpus_labeled_tp: float = 0.0
    corpus_labeled_fp: float = 0.0
    corpus_labeled_fn: float = 0.0
    corpus_unlabeled_tp: float = 0.0
    corpus_unlabeled_fp: float = 0.0
    corpus_unlabeled_fn: float = 0.0

    sent_labeled_prec_l: list[float] = []
    sent_labeled_recall_l: list[float] = []
    sent_labeled_f1_l: list[float] = []
    sent_unlabeled_prec_l: list[float] = []
    sent_unlabeled_recall_l: list[float] = []
    sent_unlabeled_f1_l: list[float] = []

    for sent in dataset.sents:
        pred_tree_str = dataset.action_dict.build_tree_str(action_ids=sent.best_actions, tokens=sent.tokens_w_eos)

        # Calculate labeled scores.
        # Conversion to span can be done even if the tree_str does not form a tree.
        pred_spans = tree_to_span(tree_str=pred_tree_str)
        gold_spans = tree_to_span(tree_str=sent.gold_tree_str_w_eos)

        tp, fp, fn = utils.calc_parsing_scores(pred_spans=pred_spans, gold_spans=gold_spans)

        # Calculate unlabeled scores.
        unlabeled_pred_spans = utils.get_unlabeled_spans(span_count=pred_spans)
        unlabeled_gold_spans = utils.get_unlabeled_spans(span_count=gold_spans)

        unlabeled_tp, unlabeled_fp, unlabeled_fn = utils.calc_parsing_scores(
            pred_spans=unlabeled_pred_spans, gold_spans=unlabeled_gold_spans
        )
        prec, recall, f1 = utils.calc_prec_recall_f1(tp=tp, fp=fp, fn=fn)
        unlabeled_prec, unlabeled_recall, unlabeled_f1 = utils.calc_prec_recall_f1(
            tp=unlabeled_tp, fp=unlabeled_fp, fn=unlabeled_fn
        )

        sent_labeled_prec_l.append(prec)
        sent_labeled_recall_l.append(recall)
        sent_labeled_f1_l.append(f1)
        corpus_labeled_tp += tp
        corpus_labeled_fp += fp
        corpus_labeled_fn += fn

        sent_unlabeled_prec_l.append(unlabeled_prec)
        sent_unlabeled_recall_l.append(unlabeled_recall)
        sent_unlabeled_f1_l.append(unlabeled_f1)
        corpus_unlabeled_tp += unlabeled_tp
        corpus_unlabeled_fp += unlabeled_fp
        corpus_unlabeled_fn += unlabeled_fn

    labeled_corpus_prec, labeled_corpus_recall, labeled_corpus_f1 = utils.calc_prec_recall_f1(
        tp=corpus_labeled_tp, fp=corpus_labeled_fp, fn=corpus_labeled_fn
    )
    unlabeled_corpus_prec, unlabeled_corpus_recall, unlabeled_corpus_f1 = utils.calc_prec_recall_f1(
        tp=corpus_unlabeled_tp, fp=corpus_unlabeled_fp, fn=corpus_unlabeled_fn
    )

    labeled_sent_f1: float = np.mean(sent_labeled_f1_l)
    labeled_sent_prec: float = np.mean(sent_labeled_prec_l)
    labeled_sent_recall: float = np.mean(sent_labeled_recall_l)
    unlabeled_sent_f1: float = np.mean(sent_unlabeled_f1_l)
    unlabeled_sent_prec: float = np.mean(sent_unlabeled_prec_l)
    unlabeled_sent_recall: float = np.mean(sent_unlabeled_recall_l)

    return (
        labeled_corpus_prec,
        labeled_corpus_recall,
        labeled_corpus_f1,
        unlabeled_corpus_prec,
        unlabeled_corpus_recall,
        unlabeled_corpus_f1,
        labeled_sent_prec,
        labeled_sent_recall,
        labeled_sent_f1,
        unlabeled_sent_prec,
        unlabeled_sent_recall,
        unlabeled_sent_f1,
    )


@dataclass
class ActionInfo:
    is_nt: bool
    stack_top_action_i: int | None
    bottomup_id: int | None  # Node id calculated by bottom-up traversal.
    child_ids: list[int] | None  # Bottom-up id of the childs.


# The algorithm is naive and maybe inefficient.
def calc_sent_nt_insert_pos(
    action_ids: list[int],
    action_dict: GeneralizedActionDict,
) -> dict[str, int]:
    # Prepare stack to just store the action indices and whether the action is nt.
    stack: list[tuple[int, bool]] = []
    # Store information for each action: (is action nt, stack top action id, right-most leaf id, child right-most leaf ids)
    # Note that right-most leaf id and child right-most leaf ids cannot be determined for nts until it's reduced; so simply set None for these values.
    action_res: list[ActionInfo] = []

    bottom_up_count: int = 0
    leaf_count: int = 0

    for action_i, a_id in enumerate(action_ids):
        if action_dict.is_nt(a_id):
            # First update action_res.
            nt_pos = action_dict.nt_pos(a_id)
            stack_top_action_i = stack[-1][0] if nt_pos != 0 else None
            # bottomup_id, child_ids, cvhild_subw are updated later.
            action_res.append(
                ActionInfo(
                    is_nt=True,
                    stack_top_action_i=stack_top_action_i,
                    bottomup_id=None,
                    child_ids=None,
                )
            )

            # Next, update stack.
            comp_node_count = 0
            for pos in range(len(stack), -1, -1):
                if pos < len(stack) and not stack[pos][1]:
                    comp_node_count += 1

                if nt_pos == comp_node_count:
                    stack.insert(pos, (action_i, True))
                    break

        elif action_dict.is_shift(a_id):
            action_res.append(
                ActionInfo(
                    is_nt=False,
                    stack_top_action_i=None,
                    bottomup_id=bottom_up_count,
                    child_ids=[],
                )
            )
            stack.append((action_i, False))

            bottom_up_count += 1
            leaf_count += 1

        elif action_dict.is_reduce(a_id):
            # First, find open nt.
            open_idx = len(stack) - 1
            while not stack[open_idx][1]:
                open_idx -= 1

            # Next, update action_res for the nt.
            stack_top_nt_action_i = stack[open_idx][0]
            child_ids = [action_res[i[0]].bottomup_id for i in stack[open_idx + 1 :]]
            assert None not in child_ids
            # Calculate the bottom-up id for the nt.
            bottomup_id: int = child_ids[-1] + 1

            assert action_res[stack_top_nt_action_i].bottomup_id is None
            assert action_res[stack_top_nt_action_i].child_ids is None
            action_res[stack_top_nt_action_i].bottomup_id = bottomup_id
            action_res[stack_top_nt_action_i].child_ids = child_ids

            # Then, update action_res for reduced node.
            action_res.append(
                ActionInfo(
                    is_nt=False,
                    stack_top_action_i=None,
                    bottomup_id=bottomup_id,
                    child_ids=[],
                )
            )

            # Finally, update stack.
            stack = stack[:open_idx]
            stack.append((action_i, False))
            bottom_up_count += 1

    # Calculate insert position count.
    nt_insert_pos: dict[str, int] = dict()
    for action_i in range(len(action_res)):
        cur_res = action_res[action_i]
        assert cur_res.bottomup_id is not None
        assert cur_res.child_ids is not None

        if not cur_res.is_nt:
            continue

        if cur_res.stack_top_action_i is None:
            # Inserting nt on the top of stack (i.e., top down action)
            insert_pos = str(0)

        else:
            # Otherwise, nt is inserted to the left of a complete node.

            stack_top_is_nt = action_res[cur_res.stack_top_action_i].is_nt
            stack_top_bottomup_id = action_res[cur_res.stack_top_action_i].bottomup_id
            assert stack_top_bottomup_id is not None

            # Find to which child stack_top_right_most_leaf_id corresponds.
            insert_pos: str | None = None
            for j, child_id in enumerate(cur_res.child_ids):
                if stack_top_bottomup_id <= child_id:
                    if (not stack_top_is_nt) and stack_top_bottomup_id == child_id:
                        # Case when nt is just opened after j-th child is reduced.
                        if j == len(cur_res.child_ids) - 1:
                            # Bottom-up case.
                            insert_pos = "inf"
                        else:
                            insert_pos = f"{j + 1}"

                    else:
                        # Otherwise, nt is opened after only some (not all) parts of j-th child is constructed.
                        # .5 simply indicates that an nt is opened after only parts of n+1-th child is constructed.
                        insert_pos = f"{j}.5"
                    break

        # Update count.
        assert insert_pos is not None
        if insert_pos not in nt_insert_pos:
            nt_insert_pos[insert_pos] = 0

        nt_insert_pos[insert_pos] += 1

    return nt_insert_pos


def calc_nt_insert_pos(
    dataset: EvalDataset,
) -> tuple[dict[str, int], dict[str, float]]:
    """Calculate the index n for each nt action, where each nt is opened after its n-th child is completed.
    If n == 0, then the nt is opened in a top-down way; if n == 1, left-corner.
    When an nt is opened after some part of its n+1 is generated, we take the index as n + 0.5. Besides, for bottom-up nt action, inf is used as position.
    """
    corpus_nt_insert_pos: dict[str, int] = dict()
    sent_nt_insert_pos_l: list[dict[str, float]] = list()

    num_non_skipped_sents: int = 0

    for sent in dataset.sents:
        sent_poss = calc_sent_nt_insert_pos(
            action_ids=sent.best_actions,
            action_dict=dataset.action_dict,
        )
        if len(sent_poss) == 0:
            continue

        num_non_skipped_sents += 1

        for pos_key in sent_poss:
            if pos_key not in corpus_nt_insert_pos:
                corpus_nt_insert_pos[pos_key] = 0
            corpus_nt_insert_pos[pos_key] += sent_poss[pos_key]

        # Calcualte average within sentence.
        num_nt_actions = sum(sent_poss.values())
        ave_sent_pos = {k: v / num_nt_actions for k, v in sent_poss.items()}
        sent_nt_insert_pos_l.append(ave_sent_pos)

    # Calculate average over sentences.
    sent_nt_insert_pos: dict[str, float] = dict()
    for ave_sent_pos in sent_nt_insert_pos_l:
        for pos_key in ave_sent_pos:
            if pos_key not in sent_nt_insert_pos:
                sent_nt_insert_pos[pos_key] = 0

            sent_nt_insert_pos[pos_key] += ave_sent_pos[pos_key]

    ave_sent_nt_insert_pos = {k: v / num_non_skipped_sents for k, v in sent_nt_insert_pos.items()}

    return (
        corpus_nt_insert_pos,
        ave_sent_nt_insert_pos,
    )


def calc_lm_performance(dataset: EvalDataset):
    """Here, only calculates token-level ppl, not word leve"""

    all_joint_log_probs: list[float] = []

    num_tokens: int = 0

    for sent in dataset.sents:
        assert len(sent.beam_scores) == len(sent.tokens_w_eos)
        assert len(sent.beam_token_log_probs) == len(sent.tokens_w_eos)

        # eos token not removed from token counts.
        num_tokens += len(sent.tokens_w_eos)

        # Only use the last beam scores.
        # Note that the token log probs is subtracted from beam_scores when storing the beam search results (it is meaningless now though).
        beam_log_probs_except_last_token = torch.tensor(sent.beam_scores[-1])
        beam_log_probs_last_token = torch.tensor(sent.beam_token_log_probs[-1])

        assert beam_log_probs_except_last_token.size() == beam_log_probs_last_token.size()

        beam_joint_log_probs = beam_log_probs_except_last_token + beam_log_probs_last_token

        marginalized_joint_log_probs = torch.logsumexp(input=beam_joint_log_probs, dim=-1)

        all_joint_log_probs.append(marginalized_joint_log_probs.item())

    assert len(dataset.sents) == len(all_joint_log_probs)
    # Language modeling performance
    corpus_nll: float = -np.sum(all_joint_log_probs)
    # PPL marginalized over beams.
    corpus_token_ppl: float = np.exp(corpus_nll / num_tokens)

    return (
        corpus_nll,
        corpus_token_ppl,
    )


def eval_given_actions_by_running_model(
    dataset: EvalDataset, model: GeneralizedActionFixedStackRNNG, device: str, eval_gold_actions: bool
):
    with torch.no_grad():
        batch = dataset.eval_batch(eval_gold_actions=eval_gold_actions)

        token_ids, given_action_ids = batch

        token_ids = token_ids.to(device)
        given_action_ids = given_action_ids.to(device)

        try_again: bool = False
        try:
            (
                supervised_loss,
                supervised_token_loss,  # token neg ll.
                supervised_action_loss,  # action neg ll.
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
                supervised_token_loss,  # token neg ll.
                supervised_action_loss,  # action neg ll.
                supervised_token_ppl,
                supervised_action_ppl,
                nt_pos_count,
            ) = model.supervised_forward(
                x=token_ids,
                actions=given_action_ids,
                stack_size=args.larger_stack_size,  # Use default stack size first.
            )

        all_joint_neg_ll: list[float] = supervised_loss.tolist()
        all_action_neg_ll: list[float] = supervised_action_loss.tolist()
        all_token_neg_ll: list[float] = supervised_token_loss.tolist()

        num_tokens: int = sum([len(dataset.sents[i].tokens_w_eos) for i in range(len(dataset.sents))])

    # Calculate statistics.
    structure_cond_token_ppl: float = np.exp(np.sum(all_token_neg_ll) / num_tokens)

    joint_neg_ll: float = np.sum(all_joint_neg_ll)
    action_only_neg_ll: float = np.sum(all_action_neg_ll)
    token_only_neg_ll: float = np.sum(all_token_neg_ll)

    return (
        structure_cond_token_ppl,
        joint_neg_ll,
        action_only_neg_ll,
        token_only_neg_ll,
    )


def main(args):
    if args.device == "cuda":
        device = "cuda:{}".format(args.gpu)
    else:
        device = "cpu"

    logger.info(f"Using {device} as device!!")

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

    if args.filter_by_parsability:
        logger.info(f"Filter data to evaluate by pos_limit: {action_dict.nt_insert_pos_limit}")

    dataset = EvalDataset.from_files(
        inference_file=args.inference_file,
        text_file=args.text_file,
        gold_tree_file=args.gold_tree_file,
        strategy_params_file=args.strategy_params_file,
        vocab=vocab,
        action_dict=action_dict,
        strategy_key=args.strategy,
        word_sync_step_limit=args.word_sync_step_limit,
        filter_by_parsability=args.filter_by_parsability,
    )
    logger.info("model architecture")
    logger.info(model)
    model.eval()

    logger.info(f"Number of data to evaluate: {len(dataset.sents)}")

    # Evaluate parsing performance
    logger.info("Start calculating parsing scores!!")
    (
        labeled_corpus_prec,
        labeled_corpus_recall,
        labeled_corpus_f1,
        unlabeled_corpus_prec,
        unlabeled_corpus_recall,
        unlabeled_corpus_f1,
        labeled_sent_prec,
        labeled_sent_recall,
        labeled_sent_f1,
        unlabeled_sent_prec,
        unlabeled_sent_recall,
        unlabeled_sent_f1,
    ) = calc_parsing_scores(dataset=dataset)

    logger.info("Finish calculating parsing scores!!")

    # Evaluate LM performance.
    logger.info("Start calculating lm performance!!")
    (
        corpus_nll,
        corpus_token_ppl,
    ) = calc_lm_performance(dataset=dataset)
    logger.info("Finish calculating lm performance!!")

    # Evaluate nt insert positions of best action sequences.
    logger.info("Start calculating nt insert positions!!")
    (
        corpus_nt_insert_pos,
        ave_sent_nt_insert_pos,
    ) = calc_nt_insert_pos(dataset=dataset)
    logger.info("Finish calculating nt insert positions!!")

    logger.info("Start evaluation by runnign model!!")
    (
        structure_cond_token_ppl,
        joint_neg_ll,
        action_only_neg_ll,
        token_only_neg_ll,
    ) = eval_given_actions_by_running_model(dataset=dataset, model=model, device=device, eval_gold_actions=False)

    (
        gold_structure_cond_token_ppl,
        gold_joint_neg_ll,
        gold_action_only_neg_ll,
        gold_token_only_neg_ll,
    ) = eval_given_actions_by_running_model(dataset=dataset, model=model, device=device, eval_gold_actions=True)
    logger.info("Finish evaluation by runnign model!!")

    eval_results = EvalCorpusResults(
        labeled_corpus_precision=labeled_corpus_prec,
        labeled_corpus_recall=labeled_corpus_recall,
        labeled_corpus_f1=labeled_corpus_f1,
        unlabeled_corpus_precision=unlabeled_corpus_prec,
        unlabeled_corpus_recall=unlabeled_corpus_recall,
        unlabeled_corpus_f1=unlabeled_corpus_f1,
        labeled_sent_precision=labeled_sent_prec,
        labeled_sent_recall=labeled_sent_recall,
        labeled_sent_f1=labeled_sent_f1,
        unlabeled_sent_precision=unlabeled_sent_prec,
        unlabeled_sent_recall=unlabeled_sent_recall,
        unlabeled_sent_f1=unlabeled_sent_f1,
        corpus_nll=corpus_nll,
        corpus_token_ppl=corpus_token_ppl,
        corpus_nt_insert_pos=corpus_nt_insert_pos,
        sent_nt_insert_pos=ave_sent_nt_insert_pos,
        structure_cond_token_ppl=structure_cond_token_ppl,
        gold_structure_cond_token_ppl=gold_structure_cond_token_ppl,
        joint_neg_ll=joint_neg_ll,
        action_only_neg_ll=action_only_neg_ll,
        token_only_neg_ll=token_only_neg_ll,
        gold_joint_neg_ll=gold_joint_neg_ll,
        gold_action_only_neg_ll=gold_action_only_neg_ll,
        gold_token_only_neg_ll=gold_token_only_neg_ll,
    )

    # Write results to file.
    with open(args.output_file, mode="w") as f:
        json.dump(dataclasses.asdict(eval_results), f, indent="\t")

    logger.info("Finished all evaluation and saved the results!!")


if __name__ == "__main__":
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s:%(name)s:%(levelname)s: %(message)s",
    )

    main(args)
