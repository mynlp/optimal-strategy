import random
import re
from typing import Any, TypeAlias
from venv import logger  # TODO: use custom logger for safety.

import nltk
import numpy as np
import torch
from action_dict import GeneralizedActionDict, NTInsertPosLimitError
from nltk import Tree
from vocab import SentencePieceVocabulary


def wrap_token_with_eos(tokens: list[str], spvocab: SentencePieceVocabulary) -> list[str]:
    return tokens + [spvocab.eos]


def wrap_tree_str_with_eos(tree_str: str, spvocab: SentencePieceVocabulary) -> str:
    return tree_str + f" {spvocab.eos}"


def filter_advanced_index(advanced_index: tuple[torch.Tensor, ...], filter: torch.Tensor) -> tuple[torch.Tensor, ...]:
    filtered_index: tuple[torch.Tensor, ...] = ()

    for dim_tensor in advanced_index:
        filtered_index += (dim_tensor[filter],)

    return filtered_index


def print_grad(model: torch.nn.Module):
    num_params: int = len(list(model.parameters()))
    grad_norm_max = torch.tensor([torch.abs(p.grad).max() for p in model.parameters() if p.grad is not None]).max()
    # Only examine nonzero grads.
    grad_norm_zero = [
        name
        for name, param in model.named_parameters()
        if param.grad is not None and torch.abs(param.grad[param.grad > 0]).numel() == 0
    ]
    if len(grad_norm_zero) > 0:
        logger.info(f"zero grad params: {len(grad_norm_zero)}/{num_params}")

    if len(grad_norm_zero) < num_params:
        grad_norm_min = torch.tensor(
            [
                torch.abs(p.grad[p.grad > 0]).min()
                for p in model.parameters()
                if p.grad is not None and p.grad[p.grad > 0].numel() > 0
            ]
        ).min()
    else:
        grad_norm_min = 0.0
    logger.info(f"Gradient: min {grad_norm_min}, max {grad_norm_max}")


def update_nt_pos_count(nt_pos_count: dict[int, int], new_counts: list[int]) -> dict[int, int]:
    for pos, count in enumerate(new_counts):
        if count != 0:
            if pos not in nt_pos_count:
                nt_pos_count[pos] = 0

            nt_pos_count[pos] += count

    return nt_pos_count


def get_nt_pos_count_str(nt_pos_count: dict[int, float]) -> str:
    return "{" + ", ".join([f"{pos}:{prop:.2f}" for pos, prop in nt_pos_count.items()]) + "}"


def calculate_max_step_len(actions: list[str]) -> int:
    cur_max: int = 0
    tmp_step_len: int = 0

    for a in actions:
        tmp_step_len += 1

        if a == "SHIFT":
            cur_max = max(cur_max, tmp_step_len)
            tmp_step_len = 0

    return cur_max


def get_gold_actions(
    strategy_key: str,
    strategy_params: dict[str, dict[str, Any]],
    tree_wo_eos: nltk.Tree,
    pos_limit: int | None,
    # step_limit: int | None,
) -> list[str]:
    func_key: str = strategy_params[strategy_key]["func_key"]
    params: dict[str, Any] = strategy_params[strategy_key]["params"]

    assert isinstance(func_key, str)
    assert isinstance(params, dict)

    # TODO: Implement other strategies.
    match func_key:
        case "top-down":
            assert len(params) == 0

            actions: list[str] = GeneralizedActionDict.get_topdown_actions(tree=tree_wo_eos, pos_limit=pos_limit)

        case "bottom-up":
            assert len(params) == 0

            actions: list[str] = GeneralizedActionDict.get_bottomup_actions(tree=tree_wo_eos, pos_limit=pos_limit)

        case "left-n-corner":
            assert "n" in params
            n = params["n"]

            actions: list[str] = GeneralizedActionDict.get_left_n_corner_actions(
                tree=tree_wo_eos, n=n, pos_limit=pos_limit
            )

        case "uniform-speculation":
            assert "real_pos" in params
            real_pos = params["real_pos"]

            actions: list[str] = GeneralizedActionDict.get_uniform_speculation_actions(
                tree=tree_wo_eos, real_pos=real_pos, pos_limit=pos_limit
            )

        case "local-first":
            assert "height" in params
            height = params["height"]

            actions: list[str] = GeneralizedActionDict.get_local_first_actions(
                tree=tree_wo_eos, height=height, pos_limit=pos_limit
            )

        case "global-first":
            assert "depth" in params
            depth = params["depth"]

            actions: list[str] = GeneralizedActionDict.get_global_first_actions(
                tree=tree_wo_eos, depth=depth, pos_limit=pos_limit
            )

        case _:
            raise Exception(f"No such strategy function: {func_key} (strategy_key: {strategy_key})")

    return actions


def check_if_parsable(
    tree_str_wo_eos: str, pos_limit: int | None, step_limit: int | None, strategy_params: dict[str, dict[str, Any]]
) -> bool:
    tree_wo_eos: nltk.Tree = nltk.Tree.fromstring(tree_str_wo_eos, remove_empty_top_bracketing=True)

    parsable: bool = True

    try:
        for strategy_key in strategy_params:
            actions = get_gold_actions(
                strategy_key=strategy_key, tree_wo_eos=tree_wo_eos, strategy_params=strategy_params, pos_limit=pos_limit
            )

            max_step_len = calculate_max_step_len(actions=actions)

            if step_limit is not None and max_step_len > step_limit:
                parsable = False

    except NTInsertPosLimitError:
        parsable = False

    return parsable


def set_random_seed(random_seed: int):
    """Set the random seeds to the given value."""
    random.seed(random_seed)

    torch.manual_seed(random_seed)

    np.random.seed(random_seed)


def convert_to_advanced_index(
    batch_index: tuple[torch.Tensor, ...],  # tuple of (num_data, ) size tensor.
    mask: torch.Tensor,  # (num_data, length) size tensor.
    start_idx: torch.Tensor,  # (num_data, ) size tensor.
) -> tuple[torch.Tensor, ...]:
    length = mask.size(1)

    idx_order = (
        torch.arange(length, dtype=torch.long, device=batch_index[0].device)
        .unsqueeze(0)
        # .repeat(batch_index[0].size(0), 1)
        .expand(
            batch_index[0].size(0), -1
        )  # Since in-place operation is not performed for idx_order (i.e., it's only used as reference), expand can be used instead of repeat to avoid memory copying.
    )

    flatten_mask = mask.view(-1)
    flatten_idx = (start_idx.unsqueeze(-1) + idx_order).view(-1)[flatten_mask]

    batches: tuple[torch.Tensor, ...] = ()
    for batch in batch_index:
        batches += (batch.unsqueeze(-1).expand(-1, length).reshape(-1)[flatten_mask],)

    batches += (flatten_idx,)

    return batches


def split_leaves_to_subwords(
    t: nltk.Tree,
    subword_spans: list[tuple[int, int]],  # map from original token idx to piece span idxs.
    pieces: list[str],
    current_word_id: int,
) -> tuple[nltk.Tree, int]:
    new_childs: list[str | nltk.Tree] = []
    for child in t:
        if isinstance(child, str):
            b, e = subword_spans[current_word_id]
            new_childs += pieces[b:e]
            current_word_id += 1
        else:
            # Otherwise, the child is nltk.Tree.
            transformed_child, current_word_id = split_leaves_to_subwords(child, subword_spans, pieces, current_word_id)
            new_childs.append(transformed_child)

    return nltk.Tree(t.label(), new_childs), current_word_id


def transform_to_subword_tree(tree: nltk.Tree, spvocab: "SentencePieceVocabulary") -> nltk.Tree:
    tokens: list[str] = tree.leaves()
    pieces: list[str] = spvocab.sp.encode(" ".join(tokens), out_type=str)
    end_idxs = [i + 1 for i, p in enumerate(pieces) if "▁" in p]
    begin_idxs = [0] + end_idxs[:-1]
    spans = list(zip(begin_idxs, end_idxs))  # map from original token idx to piece span idxs.

    transformed_tree, _ = split_leaves_to_subwords(t=tree, subword_spans=spans, pieces=pieces, current_word_id=0)
    return transformed_tree


def find_nts_in_tree(tree):
    tree = tree.strip()
    # return re.findall(r"(?=\(([^\s]+)\s\()", tree) # this re only works with preterminals.
    return re.findall(r"(?=\(([^\s]+)\s)", tree)  # this re should be used if preterminals are already removed.


SpanCount: TypeAlias = dict[tuple[str, int, int], int]
UnlabeledSpanCount: TypeAlias = dict[tuple[int, int], int]


def calc_labeled_f1(pred_spans: SpanCount, gold_spans: SpanCount) -> float:
    """This function is simplified to used during training."""
    tp = sum(
        [min(pred_spans[span], gold_spans[span]) for span in gold_spans if span in pred_spans]
        + [0]  # Just to avoid sum([]).
    )
    num_pred_spans = sum(pred_spans.values())
    num_gold_spans = sum(gold_spans.values())

    # Return 1.0 precision/recall when there are no spans to predict/predicted.
    # Note that in case num_pred_spans == 0 && num_gold_spans > 0, the resulting f1 will be 0.0.
    prec = tp / num_pred_spans if num_pred_spans > 0 else 1.0
    recall = tp / num_gold_spans if num_gold_spans > 0 else 1.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0

    return f1


def calc_prec_recall_f1(tp: float, fp: float, fn: float) -> tuple[float, float, float]:
    # Return 1.0 precision/recall when there are no spans to predict/predicted.
    # Note that in case num_pred_spans == 0 && num_gold_spans > 0, the resulting f1 will be 0.0.
    prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0

    return prec, recall, f1


def calc_category_parsing_scores(
    pred_spans: SpanCount,
    gold_spans: SpanCount,
) -> dict[str, tuple[float, float, float]]:
    categories = set([cat for cat, _, _ in pred_spans.keys()]) | set([cat for cat, _, _ in gold_spans.keys()])

    scores: dict[str, tuple[float, float, float]] = dict()

    # Calculate tp, fp, fn for each category.
    for cat in categories:
        num_pred_spans = sum([count for span, count in pred_spans.items() if span[0] == cat])
        num_gold_spans = sum([count for span, count in gold_spans.items() if span[0] == cat])

        tp = sum(
            [min(pred_spans[span], gold_spans[span]) for span in gold_spans if span[0] == cat and span in pred_spans]
            + [0]
        )
        fp = num_pred_spans - tp
        fn = num_gold_spans - tp

        scores[cat] = (tp, fp, fn)

    return scores


def calc_parsing_scores(
    pred_spans: dict[Any, int],
    gold_spans: dict[Any, int],
) -> tuple[float, float, float]:
    tp = sum([min(pred_spans[span], gold_spans[span]) for span in gold_spans if span in pred_spans] + [0])
    num_pred_spans = sum(pred_spans.values())
    num_gold_spans = sum(gold_spans.values())
    fp = num_pred_spans - tp
    fn = num_gold_spans - tp

    return tp, fp, fn


def get_unlabeled_spans(span_count: SpanCount) -> UnlabeledSpanCount:
    unl_span_count: UnlabeledSpanCount = dict()

    for span in span_count:
        unl_span: tuple[int, int] = span[1:]

        if unl_span not in unl_span_count:
            unl_span_count[unl_span] = 0

        unl_span_count[unl_span] += span_count[span]

    return unl_span_count


def tree_to_span_sub(t: Tree, offset: int, span_count: SpanCount, ignore_dummy_top: bool) -> int:
    """Left-to-right dfs"""

    if not ignore_dummy_top:
        # Do not add the span for the dummy top node.
        span: tuple[str, int, int] = (
            t.label(),
            offset + 0,
            offset + len(t.leaves()) - 1,
        )
        if span not in span_count:
            span_count[span] = 0

        span_count[span] += 1

    for child in t:
        if isinstance(child, str):
            # Do not count leaves.
            offset += 1
        else:
            # Child is non-terminal.
            offset = tree_to_span_sub(t=child, offset=offset, span_count=span_count, ignore_dummy_top=False)

    return offset


# Span is expressed as set[tuple[str, int, int]], where tuple[int, int, int] is (nt, span left index, span right index)
def tree_to_span(tree_str: str) -> SpanCount:
    """This funciton can also be applied to non-tree forming samples.

    Note that the parentheses must be labeled and closed.
    """
    # In case tree_str does not form a tree, simply wrap it with dummy node.
    # The span for the dummy node is ignored in tree_to_span_sub.
    wrapped_tree_str: str = f"(dummy {tree_str})"

    tree = Tree.fromstring(wrapped_tree_str, remove_empty_top_bracketing=True)
    span_count: SpanCount = dict()
    tree_to_span_sub(t=tree, offset=0, span_count=span_count, ignore_dummy_top=True)
    return span_count


def pad_items(items, pad_id):
    """
    `items`: a list of lists (each row has different number of elements).

    Return:
      padded_items: a converted items where shorter rows are padded by pad_id.
      lengths: lengths of rows in original items.
    """
    lengths = [len(row) for row in items]
    max_l = max(lengths)
    for i in range(len(items)):
        items[i] = items[i] + ([pad_id] * (max_l - len(items[i])))
    return items, lengths


def get_subword_boundary_mask(tokens):
    if any("▁" in t for t in tokens):
        # subword-tokenized
        return ["▁" in t for t in tokens]
    else:
        return [True for t in tokens]
