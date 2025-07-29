#!/usr/bin/env pythListon
# -*- coding: utf-8 -*-

"""Create data files"""

import argparse
import itertools
import json
import os
import sys
from multiprocessing import Pool
from tempfile import NamedTemporaryFile
from typing import Any

import nltk
from action_dict import GeneralizedActionDict, NTInsertPosLimitError
from utils import find_nts_in_tree, transform_to_subword_tree, wrap_token_with_eos, wrap_tree_str_with_eos
from vocab import SentencePieceVocabulary


def get_sent_info(arg) -> tuple[bool, dict]:
    drop_this_data: bool = False

    orig_tree_str, setting = arg
    (
        strategy_params,
        spvocab,
        action_dict,
        apply_length_filter,
        seqlength,
        minseqlength,
    ) = setting

    assert isinstance(action_dict, GeneralizedActionDict)

    # Note that preterminals are already removed via get_treebank.py.
    orig_tree: nltk.Tree = nltk.Tree.fromstring(orig_tree_str, remove_empty_top_bracketing=True)

    # Use sentencepiece to split to subwords.
    # Note that eos is not added yet.
    tree_wo_eos = transform_to_subword_tree(tree=orig_tree, spvocab=spvocab)

    # Tokens with eos
    tokens_w_eos: list[str] = wrap_token_with_eos(tokens=tree_wo_eos.leaves(), spvocab=spvocab)

    token_ids = spvocab.get_id(tokens_w_eos)

    # Do not consider subword boundary for now.
    # is_subword_end = get_subword_boundary_mask(tokens=tokens)

    # Add eos token at the end (outside of tree).
    tree_str_w_eos = wrap_tree_str_with_eos(tree_str=tree_wo_eos.pformat(margin=sys.maxsize), spvocab=spvocab)

    # Apply sentence length filter if necessary.
    if apply_length_filter and (len(tokens_w_eos) > seqlength or len(tokens_w_eos) < minseqlength):
        drop_this_data = True

    # Calcualte the actions for fixed strategies.
    pos_limit = action_dict.nt_insert_pos_limit

    try:
        strategy_dict: dict[str, Any] = dict()

        # Calculate action sequences for the gold tree with each strategy in given strategy_params.

        action_len: int | None = None

        # strategy_params: dict[str, dict[str, Any]]: strategy_key -> {func_key:, params:}
        for strategy_key in strategy_params:
            # Just to check.
            assert strategy_key not in strategy_dict

            func_key: str = strategy_params[strategy_key]["func_key"]
            params: dict[str, Any] = strategy_params[strategy_key]["params"]

            assert isinstance(func_key, str)
            assert isinstance(params, dict)

            match func_key:
                case "top-down":
                    assert len(params) == 0

                    actions: list[str] = GeneralizedActionDict.get_topdown_actions(
                        tree=tree_wo_eos, pos_limit=pos_limit
                    )

                case "bottom-up":
                    assert len(params) == 0

                    actions: list[str] = GeneralizedActionDict.get_bottomup_actions(
                        tree=tree_wo_eos, pos_limit=pos_limit
                    )

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

            # SHIFT for eos must be added here.
            assert len([a for a in actions if a == "SHIFT"]) == len(tokens_w_eos)

            if action_len is not None:
                assert len(actions) == action_len
            else:
                action_len = len(actions)

            # Convert to ids.
            action_ids: list[int] = action_dict.to_id(actions=actions)

            # Store actions and action_ids to res_dict.
            strategy_dict[strategy_key] = {"actions": actions, "action_ids": action_ids}

        assert action_len

        res_dict: dict[str, Any] = {
            "tokens_w_eos": tokens_w_eos,
            "token_ids": token_ids,  # token_ids may have unk tokens.
            "tree_str_w_eos": tree_str_w_eos,
            "action_len": action_len,
            "strategies": strategy_dict,
        }

        return (
            drop_this_data,
            res_dict,
        )

    except NTInsertPosLimitError as e:
        print(e)
        print(
            "Required nt_insert_pos for fixed strategy exceeded nt_insert_poslimit, so this data would be dropped from the preprocessed dataset."
        )
        drop_this_data = True

        return (drop_this_data, {})


def get_tokens_lower(tree_str: str) -> list[str]:
    tree = nltk.Tree.fromstring(tree_str, remove_empty_top_bracketing=True)
    tokens = tree.leaves()
    tokens_lower = [token.lower() for token in tokens]

    return tokens, tokens_lower


def learn_sentencepiece(textfile, output_prefix, args, apply_length_filter=True) -> SentencePieceVocabulary:
    import sentencepiece as spm

    with open(textfile, "r") as f:
        trees = [tree.strip() for tree in f]
    user_defined_symbols = args.subword_user_defined_symbols or []

    with NamedTemporaryFile("wt") as tmp:
        with Pool(args.jobs) as pool:
            for sent, sent_lower in pool.map(get_tokens_lower, trees):
                if (len(sent) > args.seqlength and apply_length_filter) or len(sent) < args.minseqlength:
                    continue
                tmp.write(" ".join(sent) + "\n")
        tmp.flush()
        spm.SentencePieceTrainer.train(
            input=tmp.name,
            model_prefix=output_prefix,
            vocab_size=args.vocabsize,
            model_type=args.subword_type,
            treat_whitespace_as_suffix=True,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            user_defined_symbols=user_defined_symbols,
            normalization_rule_name="identity",  # This is necessary to avoid unexpected conversion like '（' to '('.
        )
    return SentencePieceVocabulary(sp_model_path="{}.model".format(output_prefix))


def get_data(args):
    def get_nonterminals(textfiles, jobs=-1):
        nts = set()
        for fn in textfiles:
            with open(fn, "r") as f:
                lines = [line for line in f]
            with Pool(jobs) as pool:
                local_nts = pool.map(find_nts_in_tree, lines)
                nts.update(list(itertools.chain.from_iterable(local_nts)))
        nts = sorted(list(nts))
        print("Found nonterminals: {}".format(nts))
        return nts

    def convert(
        textfile: str,
        strategy_params: dict[str, dict[str, Any]],
        seqlength: int,
        minseqlength: int,
        outfile: str,
        spvocab: SentencePieceVocabulary,
        action_dict: GeneralizedActionDict,
        apply_length_filter: bool = True,
        jobs: int = -1,
    ):
        dropped = 0
        num_sents = 0
        conv_setting = (
            strategy_params,
            spvocab,
            action_dict,
            apply_length_filter,
            seqlength,
            minseqlength,
        )

        def process_block(tree_with_settings, f):
            _dropped = 0
            with Pool(jobs) as pool:
                for drop_this_data, sent_info in pool.map(get_sent_info, tree_with_settings):
                    if drop_this_data:
                        _dropped += 1
                        continue

                    sent_info["key"] = "sentence"
                    f.write(json.dumps(sent_info) + "\n")
            return _dropped

        with open(outfile, "wt") as f, open(textfile, "r") as in_f:
            block_size = 100000
            tree_with_settings = []
            for tree in in_f:
                tree_with_settings.append((tree, conv_setting))
                if len(tree_with_settings) >= block_size:
                    dropped += process_block(tree_with_settings, f)
                    num_sents += len(tree_with_settings)
                    tree_with_settings = []
                    print(num_sents)
            if len(tree_with_settings) > 0:
                dropped += process_block(tree_with_settings, f)
                num_sents += len(tree_with_settings)

            others = {
                # "vocab": vocab.to_json_dict() if vocab is not None else None,
                "nonterminals": nonterminals,
                # "pad_token": pad,
                # "unk_token": unk,
                "args": args.__dict__,
            }
            for k, v in others.items():
                f.write(json.dumps({"key": k, "value": v}) + "\n")

        print(
            "Saved {}/{} sentences (dropped {} due to position nt_insert_poslimit or length/unk filter)".format(
                num_sents - dropped, num_sents, dropped
            )
        )

    print("First pass through data to get nonterminals...")
    nonterminals: list[str] = get_nonterminals(
        [args.trainfile, args.valfile],
        args.jobs,
    )

    print("Running sentencepiece on the training data...")
    spvocab = learn_sentencepiece(args.trainfile, args.outputfile + "-spm", args)

    action_dict = GeneralizedActionDict(
        nonterminals=nonterminals,
        nt_insert_pos_limit=args.nt_insert_pos_limit,
    )

    # Get strategy parameters.
    with open(args.strategy_params_file, mode="r") as f:
        # strategy key -> {strategy function key:, strategy parmeters:}
        strategy_params: dict[str, dict[str, Any]] = json.load(f)

    convert(
        textfile=args.valfile,
        strategy_params=strategy_params,
        seqlength=args.seqlength,
        minseqlength=args.minseqlength,
        outfile=args.outputfile + "-val.json",
        spvocab=spvocab,
        action_dict=action_dict,
        apply_length_filter=False,
        jobs=args.jobs,
    )
    convert(
        textfile=args.trainfile,
        strategy_params=strategy_params,
        seqlength=args.seqlength,
        minseqlength=args.minseqlength,
        outfile=args.outputfile + "-train.json",
        spvocab=spvocab,
        action_dict=action_dict,
        apply_length_filter=True,
        jobs=args.jobs,
    )


def main(arguments):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--nt_insert_pos_limit",
        help="Parameter to define the maximum position from stack top where nt can be inserted",
        type=int,
    )
    parser.add_argument(
        "--vocabsize",
        help="Size of vocabulary or subword vocabulary. "
        "When unkmethod is not subword, vocab is constructed "
        "by taking the top X most frequent words and "
        "rest are replaced with special UNK tokens. "
        "If unkmethod=subword, this defines the subword vocabulary size. ",
        type=int,
        default=100000,
    )
    parser.add_argument(
        "--subword_type",
        help="Segmentation algorithm in sentencepiece. Note that --treat_whitespace_as_suffix for sentence_piece is always True.",
        choices=["bpe", "unigram"],
        default="bpe",
    )
    parser.add_argument(
        "--subword_user_defined_symbols",
        nargs="*",
        help="--user_defined_symbols for sentencepiece. These tokens are not segmented into subwords.",
    )
    parser.add_argument(
        "--strategy_params_file", help="Path to json file in which strategy parameters are stored.", required=True
    )
    parser.add_argument("--trainfile", help="Path to training data.", required=True)
    parser.add_argument("--valfile", help="Path to validation data.", required=True)
    parser.add_argument(
        "--seqlength",
        help="Maximum sequence length. Sequences longer than this are dropped.",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--minseqlength",
        help="Minimum sequence length. Sequences shorter than this are dropped.",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--outputfile",
        help="Prefix of the output file names. ",
        type=str,
        required=True,
    )

    parser.add_argument("--jobs", type=int, default=-1)

    args = parser.parse_args(arguments)
    if args.jobs == -1:
        args.jobs = len(os.sched_getaffinity(0))
    get_data(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
