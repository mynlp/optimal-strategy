"""Convert treebank files into single ptb-style file."""

import argparse
import glob
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import myutils
import nltk
from mylogger import main_logger

logger = main_logger.getChild(__name__)

##########
## Util ##
##########


def remove_preterminals(t: nltk.Tree) -> nltk.Tree:
    new_childs = []
    for child in t:
        if isinstance(child, str):
            new_childs.append(child)

        elif len(child) == 1 and child.height() == 2:
            # child is preterminal.
            new_childs.append(child[0])

        else:
            # child is non-terminal but not preterminal.
            new_childs.append(remove_preterminals(child))

    return nltk.Tree(t.label(), new_childs)


def remove_prepreterminals(t: nltk.Tree) -> nltk.Tree:
    new_childs = []
    for child in t:
        if isinstance(child, str):
            new_childs.append(child)

        elif child.height() <= 3:
            # child is prepreterminal.

            # Since the tree is traversed in a top-down manner, prepreterminals are accessed before preterminals.
            assert child.height() != 2

            # This can deal with some cases where prepreterminals have more than one child.
            # This odd case happens on SPMRL Hebrew.
            new_childs.extend(child.leaves())

        else:
            # child is non-terminal but not preterminal.
            new_childs.append(remove_prepreterminals(child))

    return nltk.Tree(t.label(), new_childs)


def min_leaf_depth(t: nltk.Tree) -> int:
    leaf_depths: list[int] = [len(t.leaf_treeposition(leaf_i)) for leaf_i in range(len(t.leaves()))]
    return min(leaf_depths)


def remove_prepreterminals_spmrl_polish(t: nltk.Tree) -> nltk.Tree:
    """Remove both preterminals and lowest layer nonterminals.

    This function is necessary to deal with nested prepreterminals in SPMRL Polish.
    """
    assert t.height() > 3

    new_childs = []
    for child in t:
        assert isinstance(child, nltk.Tree)
        # Since the tree is traversed in a top-down manner, lowest layer of nonterminals are accessed before preterminals.
        # In SPMRL Polish, lowest layer of nonterminals are sometimes nested; so, we need to calculate the min_leaf depth.
        if min_leaf_depth(child) == 2:
            # child is the lowest layer of nonterminals.

            # In some cases lowest layer of nonterminals are nested and have more than one child.

            # Simply get the leaves and remove both preterminals and lowest layer of nonterminals.
            subtree_leaves: list[str] = child.leaves()

            new_childs += subtree_leaves

        else:
            assert child.height() > 3
            # child is non-terminal but not preterminal.

            new_childs.append(remove_prepreterminals_spmrl_polish(child))

    return nltk.Tree(t.label(), new_childs)


def check_null_element(t: nltk.Tree, treebank_type: str) -> bool:
    match treebank_type:
        case "ptb" | "ctb":
            return t.label() == "-NONE-"

        case "ftb":
            if len(t.leaves()) == 1:
                text = t.leaves()[0]
                if text == "*T*":
                    return True

            return False

        case "ktb" | "kortb":
            if len(t.leaves()) == 1:
                text = t.leaves()[0]

                # null elements without indexing
                if text.startswith("*") and text.endswith("*"):
                    return True
                # null elements with indexing
                if re.match("^\*(.*)\*-[0-9]+$", text):
                    return True

            return False
        case _:
            raise Exception(f"No such treebank_type: {treebank_type}")


def remove_null_element_sub(t: nltk.Tree | str, treebank_type: str) -> str:
    if isinstance(t, str):
        return t
    elif check_null_element(t, treebank_type):
        # Drop null element.
        return ""
    else:
        subtree_str_l: list[str] = []

        for child in t:
            subtree_str: str = remove_null_element_sub(child, treebank_type=treebank_type)
            subtree_str_l.append(subtree_str)

        children_str: str = " ".join(subtree_str_l)

        # Check if all the children are null elements.
        # If all children are null, simply remove the parent.
        if children_str.strip() == "":
            return ""
        else:
            tree_str: str = f"({t.label()} {children_str})"

            return tree_str


class ReturnEmptyError(Exception):
    pass


def remove_null_element(t: nltk.Tree, treebank_type: str) -> nltk.Tree:
    s = remove_null_element_sub(t, treebank_type=treebank_type)
    if s == "":
        # KTB has a tree that consists of only null elements...
        raise ReturnEmptyError
    else:
        return nltk.Tree.fromstring(s)


def remove_tag_subtree_sub(t: nltk.Tree | str, tag: str) -> str:
    if isinstance(t, str):
        return t
    elif t.label() == tag:
        # Drop the tag subtree.
        return ""
    else:
        subtree_str_l: list[str] = []

        for child in t:
            subtree_str: str = remove_tag_subtree_sub(child, tag=tag)
            subtree_str_l.append(subtree_str)

        children_str: str = " ".join(subtree_str_l)

        # Check if all the children are removed.
        # If all children are removed, simply remove the parent.
        if children_str.strip() == "":
            return ""
        else:
            tree_str: str = f"({t.label()} {children_str})"

            return tree_str


def remove_tag_subtree(t: nltk.Tree, tag: str) -> nltk.Tree:
    s = remove_tag_subtree_sub(t, tag=tag)
    if s == "":
        raise ReturnEmptyError
    else:
        return nltk.Tree.fromstring(s)


class PreterminalRootError(Exception):
    pass


def normalize_cateogry(cat_str: str) -> str:
    if cat_str.startswith("-") and cat_str.endswith("-"):
        # Leave categories, e.g., -LRB-
        return cat_str

    # Get the left side of '-'
    cat: str = cat_str.split("-")[0]

    # Get the left side of '='
    cat: str = cat.split("=")[0]

    # Get the left side of '|'
    cat: str = cat.split("|")[0]

    # Get the left side of ';'
    # Spcific to Keyaki Treebank.
    cat: str = cat.split(";")[0]

    # Get the left side of '{'
    # Spcific to Keyaki Treebank.
    cat: str = cat.split("{")[0]

    # Specific to SPMRL
    cat: str = cat.split("##")[0]

    return cat


def normalize_cat_tree(t: nltk.Tree) -> None:
    cur_label: str = t.label()
    # Set normalized label.
    t.set_label(label=normalize_cateogry(cur_label))

    for child in t:
        if isinstance(child, str):
            # Do nothing.
            continue
        else:
            # Otherwise, the child is nltk.Tree.
            normalize_cat_tree(child)

    return


############################
## Treebank Load Function ##
############################


def get_ptb_data(dirpath: Path, out_train: Path, out_dev_22: Path, out_dev_24: Path, out_test: Path):
    # Set sections.
    train_sections = [f"{n:0=2}" for n in range(2, 22)]
    test_sections = ["23"]
    dev_22_sections = ["22"]
    dev_24_sections = ["24"]

    split_sections = [train_sections, test_sections, dev_22_sections, dev_24_sections]
    output_filepaths = [out_train, out_test, out_dev_22, out_dev_24]

    # Iterate over the train-dev-test splits.
    for sections, output_filepath in zip(split_sections, output_filepaths):
        t_str_l: list[str] = []

        for section in sections:
            file_count = 0

            for file_path in glob.glob(str(dirpath.joinpath(section, "*"))):
                file_count += 1

                with Path(file_path).open(mode="r") as f:
                    cur_str: str = ""
                    num_left_bracket: int = 0
                    num_right_bracket: int = 0

                    for l in f:
                        if l == "\n":
                            continue
                        else:
                            cur_str += l
                            num_left_bracket += l.count("(")
                            num_right_bracket += l.count(")")

                            if num_left_bracket == num_right_bracket:
                                t = nltk.Tree.fromstring(cur_str, remove_empty_top_bracketing=True)

                                # Remove null element.
                                t = remove_null_element(t, "ptb")

                                # Normailze category.
                                normalize_cat_tree(t)

                                # Remove preterminals.
                                t = remove_preterminals(t=t)

                                t_str: str = t.pformat(margin=sys.maxsize)

                                t_str_l.append(t_str)

                                cur_str = ""
                                num_left_bracket = 0
                                num_right_bracket = 0
            if file_count <= 0:
                raise Exception(f"No file found in section {section}")

        with output_filepath.open(mode="w") as g:
            g.write("\n".join(t_str_l))


def get_ctb_data(
    dirpath: Path,
    out_train: Path,
    out_dev: Path,
    out_test: Path,
):
    # Set sections.
    train_sections = list(range(1, 270 + 1)) + list(range(440, 1151 + 1))
    dev_sections = list(range(301, 325 + 1))
    test_sections = list(range(271, 300 + 1))

    split_sections = [train_sections, dev_sections, test_sections]
    output_filepaths = [out_train, out_dev, out_test]

    # Iterate over the train-dev-test splits.
    for sections, output_filepath in zip(split_sections, output_filepaths):
        t_str_l: list[str] = []
        for section in sections:
            section_file = dirpath.joinpath(f"chtb_{section:0=3}.fid")

            # Simply skip if there is not corresponding file for the fid.
            if not section_file.exists():
                logger.info(f"Skipt fid {section}; there is no corresponding file.")
                continue

            with section_file.open(mode="r", encoding="GB2312") as f:
                cur_str: str = ""
                num_left_bracket: int = 0
                num_right_bracket: int = 0

                for l in f:
                    if len(l.lstrip()) == 0:
                        continue
                    elif l.lstrip()[0] == "<":
                        continue
                    else:
                        cur_str += l
                        num_left_bracket += l.count("(")
                        num_right_bracket += l.count(")")

                        if num_left_bracket == num_right_bracket:
                            t = nltk.Tree.fromstring(cur_str, remove_empty_top_bracketing=True)

                            # Remove null element.
                            t = remove_null_element(t, "ctb")

                            # Normailze category.
                            normalize_cat_tree(t)

                            # Remove preterminals.
                            t = remove_preterminals(t=t)

                            t_str: str = t.pformat(margin=sys.maxsize)

                            t_str_l.append(t_str)

                            cur_str = ""
                            num_left_bracket = 0
                            num_right_bracket = 0

        with output_filepath.open(mode="w") as g:
            g.write("\n".join(t_str_l))


def get_spmrl_data(
    dirpath: Path,
    out_train: Path,
    out_dev: Path,
    out_test: Path,
    lang: str,
):
    """
    The trees in SPMRL dataset do not contain null elements.
    Args:
        lang: string key specifying the target language: Basque, French, German, Hebrew, Hungarian, Korean, Polish, swedish. Note that for Swedish somehow the initial char is not capitalized.
    """

    # Check lang key.
    match lang:
        case "Basque" | "French" | "German" | "Hebrew" | "Hungarian" | "Korean" | "Polish" | "swedish":
            pass
        case "Swedish":
            raise Exception(f"Invalid lang key: {lang}. Maybe 'swedish'...?")
        case "Polish_remove_lowest":  # Separate language key for Polish with lowest layer of nonterminals removed.
            pass
        case _:
            raise Exception(f"Invalid lang key: {lang}")

    # Use only train split.
    match lang:
        case "Basque" | "French" | "German" | "Hungarian" | "Korean" | "Polish":
            file_paths = [
                dirpath.joinpath("train", f"train.{lang}.gold.ptb"),
                dirpath.joinpath("dev", f"dev.{lang}.gold.ptb"),
                dirpath.joinpath("test", f"test.{lang}.gold.ptb"),
            ]
        case "Hebrew" | "swedish":
            # Since these two languages have only small amount of data, we use the train5k split.
            file_paths = [
                dirpath.joinpath("train5k", f"train5k.{lang}.gold.ptb"),
                dirpath.joinpath("dev", f"dev.{lang}.gold.ptb"),
                dirpath.joinpath("test", f"test.{lang}.gold.ptb"),
            ]
        case "Polish_remove_lowest":  # Use the same source data as normal Polish data.
            file_paths = [
                dirpath.joinpath("train", "train.Polish.gold.ptb"),
                dirpath.joinpath("dev", "dev.Polish.gold.ptb"),
                dirpath.joinpath("test", "test.Polish.gold.ptb"),
            ]

    output_files = [out_train, out_dev, out_test]

    for input_file, output_file in zip(file_paths, output_files):
        t_str_l: list[str] = []
        # The Files are in utf-8.
        with input_file.open(mode="r", encoding="utf-8") as f:
            # One tree per line.
            for l in f:
                if l == "\n":
                    continue
                else:
                    cur_str = l

                    # But, kortb has no top bracket.
                    t = nltk.Tree.fromstring(cur_str, remove_empty_top_bracketing=True)

                    # Process top brackets.
                    match lang:
                        # Those with non-empty top bracket
                        case "Basque" | "Hebrew" | "Korean":
                            try:
                                assert len(t) == 1
                            except:
                                logger.info(f"{input_file}")
                                logger.info(f"Error: {t}")
                                logger.info("Skip the sample.")

                            # Remove the top bracket.
                            t = t[0]

                        # Those with empty top bracket
                        case "French" | "swedish":
                            # Do nothing because the empty top brackets are already removed.
                            pass
                        # Those with no top bracket
                        case "German" | "Polish" | "Hungarian":
                            # Need not do anything.
                            pass

                    # Null elements are already removed.

                    # Normailze category.
                    normalize_cat_tree(t)

                    # Remove preterminals.
                    match lang:
                        case "Hebrew":
                            # SPMRL Hebrew has two layer preterminals.
                            # But, note that in some cases pre-preterminals are not not unary (though the reason is not clear)
                            t = remove_prepreterminals(t=t)

                        case "Polish_remove_lowest":
                            # In SPMRL Polish, the lowest layer of nonterminals behave like preterminals, but they are sometimes nested.

                            t = remove_prepreterminals_spmrl_polish(t=t)
                        case _:
                            t = remove_preterminals(t=t)

                    t_str: str = t.pformat(margin=sys.maxsize)

                    t_str_l.append(t_str)

        with output_file.open(mode="w") as g:
            g.write("\n".join(t_str_l))


@dataclass
class OptionalArgs:
    # PTB wsj
    ptb_dir: str = ""

    ptb_train_filepath: str = ""
    ptb_test_filepath: str = ""
    ptb_dev_22_filepath: str = ""
    ptb_dev_24_filepath: str = ""

    # CTB
    ctb_dir: str = ""

    ctb_train_filepath: str = ""
    ctb_dev_filepath: str = ""
    ctb_test_filepath: str = ""

    # SPMRL Basque
    spmrl_basque_dir: str = ""

    spmrl_basque_train_filepath: str = ""
    spmrl_basque_dev_filepath: str = ""
    spmrl_basque_test_filepath: str = ""

    # SPMRL French
    spmrl_french_dir: str = ""

    spmrl_french_train_filepath: str = ""
    spmrl_french_dev_filepath: str = ""
    spmrl_french_test_filepath: str = ""

    # SPMRL German
    spmrl_german_dir: str = ""

    spmrl_german_train_filepath: str = ""
    spmrl_german_dev_filepath: str = ""
    spmrl_german_test_filepath: str = ""

    # SPMRL Hebrew
    spmrl_hebrew_dir: str = ""

    spmrl_hebrew_train_filepath: str = ""
    spmrl_hebrew_dev_filepath: str = ""
    spmrl_hebrew_test_filepath: str = ""

    # SPMRL Hungarian
    spmrl_hungarian_dir: str = ""

    spmrl_hungarian_train_filepath: str = ""
    spmrl_hungarian_dev_filepath: str = ""
    spmrl_hungarian_test_filepath: str = ""

    # SPMRL Korean
    spmrl_korean_dir: str = ""

    spmrl_korean_train_filepath: str = ""
    spmrl_korean_dev_filepath: str = ""
    spmrl_korean_test_filepath: str = ""

    # SPMRL Polish
    spmrl_polish_dir: str = ""

    spmrl_polish_train_filepath: str = ""
    spmrl_polish_dev_filepath: str = ""
    spmrl_polish_test_filepath: str = ""

    # SPMRL Polish_remove_lowest
    spmrl_polish_remove_lowest_dir: str = ""

    spmrl_polish_remove_lowest_train_filepath: str = ""
    spmrl_polish_remove_lowest_dev_filepath: str = ""
    spmrl_polish_remove_lowest_test_filepath: str = ""

    # SPMRL Swedish
    spmrl_swedish_dir: str = ""

    spmrl_swedish_train_filepath: str = ""
    spmrl_swedish_dev_filepath: str = ""
    spmrl_swedish_test_filepath: str = ""


if __name__ == "__main__":
    # Set args.
    opt_parser: argparse.ArgumentParser = myutils.gen_parser(OptionalArgs, required=False)
    opt_args: OptionalArgs = OptionalArgs(**myutils.parse_and_filter(opt_parser, sys.argv[1:]))

    if (
        opt_args.ptb_train_filepath != ""
        and opt_args.ptb_test_filepath != ""
        and opt_args.ptb_dev_22_filepath != ""
        and opt_args.ptb_dev_24_filepath != ""
    ):
        logger.info("Start processing PTB!!!!")
        ptb_train_filepath: Path = Path(opt_args.ptb_train_filepath)
        ptb_train_filepath.parent.mkdir(parents=True, exist_ok=True)

        ptb_dev_22_filepath: Path = Path(opt_args.ptb_dev_22_filepath)
        ptb_dev_22_filepath.parent.mkdir(parents=True, exist_ok=True)
        ptb_dev_24_filepath: Path = Path(opt_args.ptb_dev_24_filepath)
        ptb_dev_24_filepath.parent.mkdir(parents=True, exist_ok=True)

        ptb_test_filepath: Path = Path(opt_args.ptb_test_filepath)
        ptb_test_filepath.parent.mkdir(parents=True, exist_ok=True)

        get_ptb_data(
            dirpath=Path(opt_args.ptb_dir),
            out_train=ptb_train_filepath,
            out_dev_22=ptb_dev_22_filepath,
            out_dev_24=ptb_dev_24_filepath,
            out_test=ptb_test_filepath,
        )
        logger.info("FINISH processing PTB!!!!")

    if opt_args.ctb_train_filepath != "" and opt_args.ctb_dev_filepath != "" and opt_args.ctb_test_filepath != "":
        logger.info("Start processing CTB!!!!")
        ctb_train_filepath: Path = Path(opt_args.ctb_train_filepath)
        ctb_train_filepath.parent.mkdir(parents=True, exist_ok=True)

        ctb_dev_filepath: Path = Path(opt_args.ctb_dev_filepath)
        ctb_dev_filepath.parent.mkdir(parents=True, exist_ok=True)

        ctb_test_filepath: Path = Path(opt_args.ctb_test_filepath)
        ctb_test_filepath.parent.mkdir(parents=True, exist_ok=True)

        get_ctb_data(
            dirpath=Path(opt_args.ctb_dir),
            out_train=ctb_train_filepath,
            out_dev=ctb_dev_filepath,
            out_test=ctb_test_filepath,
        )
        logger.info("FINISH processing CTB!!!!")

    if (
        opt_args.spmrl_basque_train_filepath != ""
        and opt_args.spmrl_basque_dev_filepath != ""
        and opt_args.spmrl_basque_test_filepath != ""
    ):
        logger.info("Start processing SPMRL basque!!!!")
        spmrl_basque_train_filepath: Path = Path(opt_args.spmrl_basque_train_filepath)
        spmrl_basque_train_filepath.parent.mkdir(parents=True, exist_ok=True)

        spmrl_basque_dev_filepath: Path = Path(opt_args.spmrl_basque_dev_filepath)
        spmrl_basque_dev_filepath.parent.mkdir(parents=True, exist_ok=True)

        spmrl_basque_test_filepath: Path = Path(opt_args.spmrl_basque_test_filepath)
        spmrl_basque_test_filepath.parent.mkdir(parents=True, exist_ok=True)

        get_spmrl_data(
            dirpath=Path(opt_args.spmrl_basque_dir),
            out_train=spmrl_basque_train_filepath,
            out_dev=spmrl_basque_dev_filepath,
            out_test=spmrl_basque_test_filepath,
            lang="Basque",
        )
        logger.info("FINISH processing SPMRL basque!!!!")

    if (
        opt_args.spmrl_french_train_filepath != ""
        and opt_args.spmrl_french_dev_filepath != ""
        and opt_args.spmrl_french_test_filepath != ""
    ):
        logger.info("Start processing SPMRL french!!!!")
        spmrl_french_train_filepath: Path = Path(opt_args.spmrl_french_train_filepath)
        spmrl_french_train_filepath.parent.mkdir(parents=True, exist_ok=True)

        spmrl_french_dev_filepath: Path = Path(opt_args.spmrl_french_dev_filepath)
        spmrl_french_dev_filepath.parent.mkdir(parents=True, exist_ok=True)

        spmrl_french_test_filepath: Path = Path(opt_args.spmrl_french_test_filepath)
        spmrl_french_test_filepath.parent.mkdir(parents=True, exist_ok=True)

        get_spmrl_data(
            dirpath=Path(opt_args.spmrl_french_dir),
            out_train=spmrl_french_train_filepath,
            out_dev=spmrl_french_dev_filepath,
            out_test=spmrl_french_test_filepath,
            lang="French",
        )
        logger.info("FINISH processing SPMRL french!!!!")

    if (
        opt_args.spmrl_german_train_filepath != ""
        and opt_args.spmrl_german_dev_filepath != ""
        and opt_args.spmrl_german_test_filepath != ""
    ):
        logger.info("Start processing SPMRL german!!!!")
        spmrl_german_train_filepath: Path = Path(opt_args.spmrl_german_train_filepath)
        spmrl_german_train_filepath.parent.mkdir(parents=True, exist_ok=True)

        spmrl_german_dev_filepath: Path = Path(opt_args.spmrl_german_dev_filepath)
        spmrl_german_dev_filepath.parent.mkdir(parents=True, exist_ok=True)

        spmrl_german_test_filepath: Path = Path(opt_args.spmrl_german_test_filepath)
        spmrl_german_test_filepath.parent.mkdir(parents=True, exist_ok=True)

        get_spmrl_data(
            dirpath=Path(opt_args.spmrl_german_dir),
            out_train=spmrl_german_train_filepath,
            out_dev=spmrl_german_dev_filepath,
            out_test=spmrl_german_test_filepath,
            lang="German",
        )
        logger.info("FINISH processing SPMRL german!!!!")

    if (
        opt_args.spmrl_hebrew_train_filepath != ""
        and opt_args.spmrl_hebrew_dev_filepath != ""
        and opt_args.spmrl_hebrew_test_filepath != ""
    ):
        logger.info("Start processing SPMRL hebrew!!!!")
        spmrl_hebrew_train_filepath: Path = Path(opt_args.spmrl_hebrew_train_filepath)
        spmrl_hebrew_train_filepath.parent.mkdir(parents=True, exist_ok=True)

        spmrl_hebrew_dev_filepath: Path = Path(opt_args.spmrl_hebrew_dev_filepath)
        spmrl_hebrew_dev_filepath.parent.mkdir(parents=True, exist_ok=True)

        spmrl_hebrew_test_filepath: Path = Path(opt_args.spmrl_hebrew_test_filepath)
        spmrl_hebrew_test_filepath.parent.mkdir(parents=True, exist_ok=True)

        get_spmrl_data(
            dirpath=Path(opt_args.spmrl_hebrew_dir),
            out_train=spmrl_hebrew_train_filepath,
            out_dev=spmrl_hebrew_dev_filepath,
            out_test=spmrl_hebrew_test_filepath,
            lang="Hebrew",
        )
        logger.info("FINISH processing SPMRL hebrew!!!!")

    if (
        opt_args.spmrl_hungarian_train_filepath != ""
        and opt_args.spmrl_hungarian_dev_filepath != ""
        and opt_args.spmrl_hungarian_test_filepath != ""
    ):
        logger.info("Start processing SPMRL hungarian!!!!")
        spmrl_hungarian_train_filepath: Path = Path(opt_args.spmrl_hungarian_train_filepath)
        spmrl_hungarian_train_filepath.parent.mkdir(parents=True, exist_ok=True)

        spmrl_hungarian_dev_filepath: Path = Path(opt_args.spmrl_hungarian_dev_filepath)
        spmrl_hungarian_dev_filepath.parent.mkdir(parents=True, exist_ok=True)

        spmrl_hungarian_test_filepath: Path = Path(opt_args.spmrl_hungarian_test_filepath)
        spmrl_hungarian_test_filepath.parent.mkdir(parents=True, exist_ok=True)

        get_spmrl_data(
            dirpath=Path(opt_args.spmrl_hungarian_dir),
            out_train=spmrl_hungarian_train_filepath,
            out_dev=spmrl_hungarian_dev_filepath,
            out_test=spmrl_hungarian_test_filepath,
            lang="Hungarian",
        )
        logger.info("FINISH processing SPMRL hungarian!!!!")

    if (
        opt_args.spmrl_korean_train_filepath != ""
        and opt_args.spmrl_korean_dev_filepath != ""
        and opt_args.spmrl_korean_test_filepath != ""
    ):
        logger.info("Start processing SPMRL korean!!!!")
        spmrl_korean_train_filepath: Path = Path(opt_args.spmrl_korean_train_filepath)
        spmrl_korean_train_filepath.parent.mkdir(parents=True, exist_ok=True)

        spmrl_korean_dev_filepath: Path = Path(opt_args.spmrl_korean_dev_filepath)
        spmrl_korean_dev_filepath.parent.mkdir(parents=True, exist_ok=True)

        spmrl_korean_test_filepath: Path = Path(opt_args.spmrl_korean_test_filepath)
        spmrl_korean_test_filepath.parent.mkdir(parents=True, exist_ok=True)

        get_spmrl_data(
            dirpath=Path(opt_args.spmrl_korean_dir),
            out_train=spmrl_korean_train_filepath,
            out_dev=spmrl_korean_dev_filepath,
            out_test=spmrl_korean_test_filepath,
            lang="Korean",
        )
        logger.info("FINISH processing SPMRL korean!!!!")

    if (
        opt_args.spmrl_polish_train_filepath != ""
        and opt_args.spmrl_polish_dev_filepath != ""
        and opt_args.spmrl_polish_test_filepath != ""
    ):
        logger.info("Start processing SPMRL polish!!!!")
        spmrl_polish_train_filepath: Path = Path(opt_args.spmrl_polish_train_filepath)
        spmrl_polish_train_filepath.parent.mkdir(parents=True, exist_ok=True)

        spmrl_polish_dev_filepath: Path = Path(opt_args.spmrl_polish_dev_filepath)
        spmrl_polish_dev_filepath.parent.mkdir(parents=True, exist_ok=True)

        spmrl_polish_test_filepath: Path = Path(opt_args.spmrl_polish_test_filepath)
        spmrl_polish_test_filepath.parent.mkdir(parents=True, exist_ok=True)

        get_spmrl_data(
            dirpath=Path(opt_args.spmrl_polish_dir),
            out_train=spmrl_polish_train_filepath,
            out_dev=spmrl_polish_dev_filepath,
            out_test=spmrl_polish_test_filepath,
            lang="Polish",
        )
        logger.info("FINISH processing SPMRL polish!!!!")

    if (
        opt_args.spmrl_polish_remove_lowest_train_filepath != ""
        and opt_args.spmrl_polish_remove_lowest_dev_filepath != ""
        and opt_args.spmrl_polish_remove_lowest_test_filepath != ""
    ):
        logger.info("Start processing SPMRL polish_remove_lowest!!!!")
        spmrl_polish_remove_lowest_train_filepath: Path = Path(opt_args.spmrl_polish_remove_lowest_train_filepath)
        spmrl_polish_remove_lowest_train_filepath.parent.mkdir(parents=True, exist_ok=True)

        spmrl_polish_remove_lowest_dev_filepath: Path = Path(opt_args.spmrl_polish_remove_lowest_dev_filepath)
        spmrl_polish_remove_lowest_dev_filepath.parent.mkdir(parents=True, exist_ok=True)

        spmrl_polish_remove_lowest_test_filepath: Path = Path(opt_args.spmrl_polish_remove_lowest_test_filepath)
        spmrl_polish_remove_lowest_test_filepath.parent.mkdir(parents=True, exist_ok=True)

        get_spmrl_data(
            dirpath=Path(opt_args.spmrl_polish_remove_lowest_dir),
            out_train=spmrl_polish_remove_lowest_train_filepath,
            out_dev=spmrl_polish_remove_lowest_dev_filepath,
            out_test=spmrl_polish_remove_lowest_test_filepath,
            lang="Polish_remove_lowest",
        )
        logger.info("FINISH processing SPMRL polish_remove_lowest!!!!")

    if (
        opt_args.spmrl_swedish_train_filepath != ""
        and opt_args.spmrl_swedish_dev_filepath != ""
        and opt_args.spmrl_swedish_test_filepath != ""
    ):
        logger.info("Start processing SPMRL swedish!!!!")
        spmrl_swedish_train_filepath: Path = Path(opt_args.spmrl_swedish_train_filepath)
        spmrl_swedish_train_filepath.parent.mkdir(parents=True, exist_ok=True)

        spmrl_swedish_dev_filepath: Path = Path(opt_args.spmrl_swedish_dev_filepath)
        spmrl_swedish_dev_filepath.parent.mkdir(parents=True, exist_ok=True)

        spmrl_swedish_test_filepath: Path = Path(opt_args.spmrl_swedish_test_filepath)
        spmrl_swedish_test_filepath.parent.mkdir(parents=True, exist_ok=True)

        get_spmrl_data(
            dirpath=Path(opt_args.spmrl_swedish_dir),
            out_train=spmrl_swedish_train_filepath,
            out_dev=spmrl_swedish_dev_filepath,
            out_test=spmrl_swedish_test_filepath,
            lang="swedish",
        )
        logger.info("FINISH processing SPMRL swedish!!!!")
