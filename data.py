#!/usr/bin/env python3
import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import nltk
import numpy as np
import torch
from action_dict import NTInsertPosLimitError

# from action_dict import TopDownActionDict, InOrderActionDict
from action_dict import GeneralizedActionDict
from utils import (
    check_if_parsable,
    get_gold_actions,
    pad_items,
    transform_to_subword_tree,
    wrap_token_with_eos,
    wrap_tree_str_with_eos,
)
from vocab import SentencePieceVocabulary


class Sentence(object):
    def __init__(
        self,
        tokens_w_eos: list[str],
        token_ids: list[int],  # token_ids is assumed to contain eos token.
        action_len: int,  # Force to set action_len (necessary to have batches_helper work for test cases)
        tree_str_w_eos: str | None,
        # Below are given fixed actions.
        given_action_ids: list[int] | None,
    ):
        self.tokens_w_eos = tokens_w_eos
        self.token_ids = token_ids

        self.action_len: int = action_len

        self.tree_str_w_eos = tree_str_w_eos

        self.given_action_ids = given_action_ids

    @staticmethod
    def from_json(j, strategy_key: str):
        # Get given actions.
        given_action_ids: list[int] = j["strategies"][strategy_key]["action_ids"]

        return Sentence(
            tokens_w_eos=j["tokens_w_eos"],
            token_ids=j["token_ids"],
            # Set action_len = 0 when gold tree information is not provided.
            # This is necessary for batches_helper to work for test cases.
            action_len=j.get("action_len", 0),
            tree_str_w_eos=j.get("tree_str_w_eos", None),
            given_action_ids=given_action_ids,
        )


class Dataset(object):
    def __init__(
        self,
        sents: list[Sentence],
        vocab: SentencePieceVocabulary,
        action_dict: GeneralizedActionDict,
        prepro_args: dict[str, Any],
    ):
        self.sents = sents
        self.vocab = vocab
        self.action_dict = action_dict
        self.prepro_args = prepro_args  # keeps which process is performed.

        self.vocab_size = vocab.size()

        # num_batches is calculated and set when batches_helper is executed.
        # It has to an attribute of Dataset class since batches_helper yields tensors.
        self.num_batches: int | None = None

    @staticmethod
    def from_json(
        data_file: str,
        strategy: str,
        vocab: SentencePieceVocabulary,
        action_dict: None | GeneralizedActionDict,
        # nt_insert_pos_limit: int | None = None,
        load_all_info: bool,  # If False, some info (orig_tokens, tokens, tree_str) are not loaded.
    ):
        """If vocab and action_dict are provided, they are not loaded from data_file.
        This is for sharing these across train/valid/test sents.
        """
        # Either the given action_dict or nt_insert_pos_limit must be non-None.
        # assert (action_dict or nt_insert_pos_limit) is not None

        j = Dataset._load_json_helper(path=data_file, load_all_info=load_all_info)
        action_dict = action_dict or GeneralizedActionDict(
            nonterminals=j["nonterminals"],
            # nt_insert_pos_limit=nt_insert_pos_limit,
            # Load from args for preprocess.
            nt_insert_pos_limit=j["args"]["nt_insert_pos_limit"],
        )

        sents = [Sentence.from_json(j=s, strategy_key=strategy) for s in j["sentences"]]

        return Dataset(
            sents=sents,
            vocab=vocab,
            action_dict=action_dict,
            prepro_args=j["args"],
        )

    @staticmethod
    def _load_json_helper(path: str, load_all_info: bool):
        def read_jsonl(f):
            data = {}
            sents = []
            for line in f:
                o = json.loads(line)
                k = o["key"]
                if k == "sentence":
                    if not load_all_info:
                        # Unused values are discarded here (for reducing memory for larger data).
                        # o["tokens"] = o["orig_tokens"] = o["tree_str"] = None
                        o["tokens_w_eos"] = o["tree_str_w_eos"] = None

                    sents.append(o)
                else:
                    # key except 'sentence' should only appear once
                    assert k not in data
                    data[k] = o["value"]
            data["sentences"] = sents
            return data

        try:
            with open(path) as f:
                # Old format => a single fat json object containing everything.
                return json.load(f)
        except json.decoder.JSONDecodeError:
            with open(path) as f:
                # New format => jsonl
                return read_jsonl(f)

    @staticmethod
    def from_text_file(
        text_file: str,
        vocab: SentencePieceVocabulary,
        action_dict: GeneralizedActionDict,
        prepro_args: dict[str, Any],
    ):
        sents: list[Sentence] = []
        with open(text_file) as f:
            for line in f:
                # orig_tokens is not subword splitted here.
                orig_tokens = line.strip().split()

                # TODO: refactor to use get_id.
                tokens_w_eos = vocab.sp.encode(" ".join(orig_tokens), out_type=str, add_eos=True)
                token_ids = vocab.sp.piece_to_id(tokens_w_eos)

                sent = Sentence(
                    tokens_w_eos=tokens_w_eos,
                    token_ids=token_ids,
                    action_len=0,  # Force to set action_len to 0.
                    tree_str_w_eos=None,
                    given_action_ids=None,
                )
                sents.append(sent)
        return Dataset(
            sents=sents,
            vocab=vocab,
            action_dict=action_dict,
            prepro_args=prepro_args,
        )

    def batches(
        self,
        batch_size: int,
        batch_token_size: int,
        batch_action_size: int,
        batch_group: str,
        max_length_diff: int,
        group_sentence_size: int,
        shuffle: bool = True,
    ):
        match batch_group:
            case "same_length":
                len_to_idxs = self._get_len_to_idxs()
            case "similar_length" | "similar_action_length":
                use_action_len = batch_group == "similar_action_length"
                len_to_idxs = self._get_grouped_len_to_idxs(
                    use_action_len=use_action_len,
                    max_length_diff=max_length_diff,
                    group_sentence_size=group_sentence_size,
                )
            case "random":
                len_to_idxs = self._get_random_len_to_idxs()
            case _:
                raise Exception(f"No such batch_group: {batch_group}")

        yield from self.batches_helper(
            batch_size=batch_size,
            batch_token_size=batch_token_size,
            batch_action_size=batch_action_size,
            len_to_idxs=len_to_idxs,
            shuffle=shuffle,
            test=False,
        )

    def test_batches(
        self,
        batch_size: int,
        batch_token_size: int,
        batch_action_size: int,
        block_size: int,
        group_sentence_size: int,
        max_length_diff: int,
    ):
        assert block_size > 0
        """
    Sents are first segmented (chunked) by block_size, and then, mini-batched.
    Since each batch contains batch_idx, we can recover the original order of
    data, by processing output grouping this size.

    This may be useful when outputing the parse results (approximately) streamly,
    by dumping to stdout (or file) at once for every 100~1000 sentences.
    Below is an such example to dump parse results by keeping the original sentence
    order.
    ```
    batch_size = 3
    block_size = 1000
    parses = []
    idxs = []
    for token, action, idx in dataset.test_batches(block_size):
      parses.extend(parse(token))
      idxs.extend(idx)
      if len(idxs) >= block_size:
        assert len(idxs) <= block_size
        parse_idx_to_sent_idx = sorted(list(enmearte(idxs)), key=lambda x:x[1])
        orig_order_parses = [parses[sent_idx] for (parse_idx, sent_idx) in parse_idx_to_sent_idx]
        # process orig_order_parses (e.g., print)
        parses = []
        idxs = []
    ```
    """
        for offset in range(0, len(self.sents), block_size):
            end = min(len(self.sents), offset + block_size)
            len_to_idxs = self._get_grouped_len_to_idxs(
                sent_idxs=range(offset, end),
                max_length_diff=max_length_diff,
                group_sentence_size=group_sentence_size,
            )
            yield from self.batches_helper(
                batch_size=batch_size,
                batch_token_size=batch_token_size,
                batch_action_size=batch_action_size,
                len_to_idxs=len_to_idxs,
                shuffle=False,
                test=True,
            )

    def batches_helper(
        self,
        batch_size: int,
        batch_token_size: int,
        batch_action_size: int,
        len_to_idxs: dict[int, list[int]],
        shuffle: bool = True,
        test: bool = False,
    ):
        # `len_to_idxs` summarizes sentence length to idx in `self.sents`.
        # This may be a subset of sentences, or full sentences.
        batches = []
        for length, idxs in len_to_idxs.items():
            if shuffle:
                idxs = np.random.permutation(idxs)

            def add_batch(begin, end):
                assert begin < end
                batches.append(idxs[begin:end])

            longest_sent_len = 0
            longest_action_len = 0
            batch_i = 0  # for i-th batch
            b = 0
            tmp_batch_token_size = batch_token_size
            tmp_batch_action_size = batch_action_size
            # Create each batch to guarantee that (batch_size*max_sent_len) does not exceed
            # batch_token_size.
            for i in range(len(idxs)):
                cur_sent_len = len(self.sents[idxs[i]].token_ids)
                cur_action_len = self.sents[idxs[i]].action_len
                longest_sent_len = max(longest_sent_len, cur_sent_len)
                longest_action_len = max(longest_action_len, cur_action_len)
                if len(self.sents[idxs[i]].token_ids) > 100:
                    # Long sequence often requires larger memory and tend to cause memory error.
                    # Here we try to reduce the elements in a batch for such sequences, considering
                    # that they are rare and will not affect the total speed much.
                    tmp_batch_token_size = int(batch_token_size * 0.7)
                    tmp_batch_action_size = int(batch_action_size * 0.7)
                if i > b and (  # for ensuring batch size 1
                    (longest_sent_len * (batch_i + 1) >= tmp_batch_token_size)
                    or (longest_action_len * (batch_i + 1) >= tmp_batch_action_size)
                    # or (batch_i > 0 and batch_i % self.batch_size == 0)
                    or (batch_i > 0 and batch_i % batch_size == 0)
                ):
                    add_batch(b, i)
                    batch_i = 0  # i is not included in prev batch
                    longest_sent_len = cur_sent_len
                    longest_action_len = cur_action_len
                    b = i
                    # batch_token_size = self.batch_token_size
                    # batch_action_size = self.batch_action_size
                    tmp_batch_token_size = batch_token_size
                    tmp_batch_action_size = batch_action_size
                batch_i += 1
            add_batch(b, i + 1)
        self.num_batches = len(batches)

        if shuffle:
            # This doesn't work due to error related to inhomogeneous shapes.
            # batches = np.random.permutation(batches)
            random.shuffle(batches)

        for batch_idx in batches:
            token_ids = [self.sents[i].token_ids for i in batch_idx]
            tokens = torch.tensor(self._pad_token_ids(token_ids), dtype=torch.long)
            ret = (tokens,)

            # is_subword_ends = [self.sents[i].is_subword_end for i in batch_idx]
            # ret += (torch.tensor(pad_items(is_subword_ends, 0)[0], dtype=torch.bool),)

            # if not test and strategy is not None:
            if not test:
                given_action_ids = [self.sents[i].given_action_ids for i in batch_idx]
                ret += (
                    torch.tensor(
                        self._pad_action_ids(action_ids=given_action_ids),
                        dtype=torch.long,
                    ),
                )

            ret += (batch_idx,)
            yield ret

    def _get_len_to_idxs(self, sent_idxs=[]):
        def to_len(token_ids):
            return len(token_ids)

        return self._get_len_to_idxs_helper(to_len, sent_idxs)

    def _get_grouped_len_to_idxs(
        self,
        group_sentence_size: int,
        sent_idxs=[],
        use_action_len=False,
        max_length_diff=20,
    ):
        if use_action_len:

            def get_length(sent: Sentence):
                return sent.action_len

        else:

            def get_length(sent: Sentence):
                return len(sent.token_ids)

        if len(sent_idxs) == 0:
            sent_idxs = range(len(self.sents))
        len_to_idxs = defaultdict(list)
        # group_size = self.group_sentence_size
        group_size = group_sentence_size
        sent_idxs_with_len = sorted([(i, get_length(self.sents[i])) for i in sent_idxs], key=lambda x: x[1])
        b = 0
        last_idx = 0
        while b < len(sent_idxs_with_len):
            min_len = sent_idxs_with_len[b][1]
            max_len = sent_idxs_with_len[min(b + group_size, len(sent_idxs_with_len) - 1)][1]
            if max_len - min_len < max_length_diff:  # small difference in a group -> regist as a group
                group = [i for i, l in sent_idxs_with_len[b : b + group_size]]
                b += group_size
            else:
                e = b + 1
                while e < len(sent_idxs_with_len) and sent_idxs_with_len[e][1] - min_len < max_length_diff:
                    e += 1
                group = [i for i, l in sent_idxs_with_len[b:e]]
                b = e
            len_to_idxs[get_length(self.sents[group[-1]])] += group
        return len_to_idxs

    def _get_random_len_to_idxs(self, sent_idxs=[]):
        def to_len(token_ids):
            return 1  # all sentences belong to the same group

        return self._get_len_to_idxs_helper(to_len, sent_idxs)

    def _get_len_to_idxs_helper(self, calc_len, sent_idxs=[]):
        if len(sent_idxs) == 0:
            sent_idxs = range(len(self.sents))
        len_to_idxs = defaultdict(list)
        for i in sent_idxs:
            sent = self.sents[i]
            len_to_idxs[calc_len(sent.token_ids)].append(i)
        return len_to_idxs

    def _pad_action_ids(self, action_ids):
        action_ids, _ = pad_items(action_ids, self.action_dict.padding_idx)
        return action_ids

    def _pad_token_ids(self, token_ids):
        token_ids, _ = pad_items(token_ids, self.vocab.padding_idx)
        return token_ids


@dataclass
class EvalSentence:
    tokens_w_eos: list[str]
    token_ids: list[int]  # eos included.
    best_actions: list[int]
    beam_scores: list[list[float]]  # Token log probs are subtracted.
    beam_token_log_probs: list[list[float]]
    gold_tree_str_w_eos: str
    gold_actions: list[int]


class EvalDataset:
    def __init__(
        self,
        sents: list[EvalSentence],
        vocab: SentencePieceVocabulary,
        action_dict: GeneralizedActionDict,
    ):
        # Data used for evaluation.
        self.sents = sents

        self.vocab = vocab
        self.action_dict = action_dict

        self.vocab_size = vocab.size()

    @staticmethod
    def from_files(
        inference_file: str,
        text_file: str,
        gold_tree_file: str,
        strategy_params_file: str,
        vocab: SentencePieceVocabulary,
        action_dict: GeneralizedActionDict,
        strategy_key: str,
        word_sync_step_limit: int,
        filter_by_parsability: bool,  # Only evaluate data that are parsable with given pos_limit and step_limit.
    ) -> "EvalDataset":
        # First, load the tokens.
        tokens_w_eos: list[list[str]] = []
        token_ids: list[list[int]] = []
        with open(text_file, mode="r") as f:
            for line in f:
                orig_tokens_line = line.strip().split()
                tokens_line: list[str] = wrap_token_with_eos(
                    tokens=vocab.sp.encode(" ".join(orig_tokens_line), out_type=str), spvocab=vocab
                )
                token_ids_line: list[int] = vocab.sp.piece_to_id(tokens_line)

                tokens_w_eos.append(tokens_line)
                token_ids.append(token_ids_line)

        # Load strategy parameters.
        with open(strategy_params_file, mode="r") as f:
            # strategy key -> {strategy function key:, strategy parmeters:}
            strategy_params: dict[str, dict[str, Any]] = json.load(f)

        # Next, load gold trees.
        gold_tree_strs: list[str] = []  # With EOS.
        gold_tree_strs_wo_eos: list[str] = []  # Without EOS; This is necessary to check parsability.
        gold_actions_l: list[list[str]] = []  # With SHIFT for EOS.
        with open(gold_tree_file, mode="r") as g:
            for line in g:
                orig_tree_str = line.strip()

                orig_tree: nltk.Tree = nltk.Tree.fromstring(orig_tree_str, remove_empty_top_bracketing=True)
                tree_wo_eos = transform_to_subword_tree(tree=orig_tree, spvocab=vocab)

                gold_actions = get_gold_actions(
                    strategy_key=strategy_key,
                    strategy_params=strategy_params,
                    tree_wo_eos=tree_wo_eos,
                    # Do not set pos limit here.
                    # pos_limit filter would be applied later.
                    pos_limit=None,
                )
                gold_actions_l.append(gold_actions)

                tree_str_wo_eos = tree_wo_eos.pformat(margin=sys.maxsize)

                tree_str_w_eos = wrap_tree_str_with_eos(tree_str=tree_str_wo_eos, spvocab=vocab)

                gold_tree_strs_wo_eos.append(tree_str_wo_eos)
                gold_tree_strs.append(tree_str_w_eos)

        # Finally, load beam search results.

        best_actions: list[list[int]] = []
        beam_scores: list[list[list[float]]] = []
        beam_token_log_probs: list[list[list[float]]] = []
        with open(inference_file, mode="r") as h:
            for line in h:
                res = json.loads(line)
                best_actions.append(res["best_actions"])
                beam_scores.append(res["beam_scores"])
                beam_token_log_probs.append(res["beam_token_log_probs"])

        # Check the number of data
        assert len(tokens_w_eos) == len(token_ids)
        assert len(tokens_w_eos) == len(best_actions)
        assert len(tokens_w_eos) == len(gold_tree_strs)

        sents: list[EvalSentence] = []
        for tok, tok_ids, best_acts, b_scores, b_tok_log_prob, gold_tree, gold_tree_wo_eos, gold_actions in zip(
            tokens_w_eos,
            token_ids,
            best_actions,
            beam_scores,
            beam_token_log_probs,
            gold_tree_strs,
            gold_tree_strs_wo_eos,
            gold_actions_l,
        ):
            # Only add sentences that are parsable with given nt_insert_pos_limit and word_sync_step_limit.
            if filter_by_parsability and not check_if_parsable(
                tree_str_wo_eos=gold_tree_wo_eos,
                pos_limit=action_dict.nt_insert_pos_limit,
                step_limit=word_sync_step_limit,
                strategy_params=strategy_params,
            ):
                continue

            # Still, we need to take care of pos_limit anyway to convert to action_ids.
            try:
                gold_action_ids = action_dict.to_id(actions=gold_actions)
            except NTInsertPosLimitError:
                continue

            sent: EvalSentence = EvalSentence(
                tokens_w_eos=tok,
                token_ids=tok_ids,
                best_actions=best_acts,
                beam_scores=b_scores,
                beam_token_log_probs=b_tok_log_prob,
                gold_tree_str_w_eos=gold_tree,
                gold_actions=gold_action_ids,
            )
            sents.append(sent)

        return EvalDataset(
            sents=sents,
            vocab=vocab,
            action_dict=action_dict,
        )

    def eval_batch(self, eval_gold_actions: bool):
        # Ad hoc: Put all data into one batch.

        num_sents: int = len(self.sents)

        token_ids = [self.sents[i].token_ids for i in range(num_sents)]
        token_ids_tensor = torch.tensor(self._pad_token_ids(token_ids), dtype=torch.long)

        if not eval_gold_actions:
            action_ids = [self.sents[i].best_actions for i in range(num_sents)]
        else:
            action_ids = [self.sents[i].gold_actions for i in range(num_sents)]

        action_ids_tensor = torch.tensor(self._pad_action_ids(action_ids=action_ids), dtype=torch.long)

        ret = (token_ids_tensor, action_ids_tensor)

        return ret

    def _pad_action_ids(self, action_ids):
        action_ids, _ = pad_items(action_ids, self.action_dict.padding_idx)
        return action_ids

    def _pad_token_ids(self, token_ids):
        token_ids, _ = pad_items(token_ids, self.vocab.padding_idx)
        return token_ids


@dataclass
class EvalCorpusResults:
    # Parsing performance
    labeled_corpus_precision: float
    labeled_corpus_recall: float
    labeled_corpus_f1: float
    unlabeled_corpus_precision: float
    unlabeled_corpus_recall: float
    unlabeled_corpus_f1: float
    labeled_sent_precision: float
    labeled_sent_recall: float
    labeled_sent_f1: float
    unlabeled_sent_precision: float
    unlabeled_sent_recall: float
    unlabeled_sent_f1: float

    # Strategy features (of best action sequence)
    corpus_nt_insert_pos: dict[str, int]  # insert position -> count
    sent_nt_insert_pos: dict[str, float]  # insert position -> average of propotion within sentence

    # Probability over sentences.
    corpus_nll: float  # Negative log likelihood.
    # PPL marginalized over beams.
    corpus_token_ppl: float

    # Structure-conditioned token probability of the best action and gold action.
    # Corpus level ppl.
    structure_cond_token_ppl: float
    gold_structure_cond_token_ppl: float

    joint_neg_ll: float
    action_only_neg_ll: float
    token_only_neg_ll: float

    gold_joint_neg_ll: float
    gold_action_only_neg_ll: float
    gold_token_only_neg_ll: float


@dataclass
class SupervisedValResult:
    # Train info
    epoch: int
    step: int

    nt_pos_count: dict[int, float]

    # Supervised validation losses.
    supervised_loss: float
    supervised_action_loss: float
    supervised_token_loss: float
    # supervised_ppl: float | None
    supervised_action_ppl: float
    supervised_token_ppl: float
