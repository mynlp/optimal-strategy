"""Convert a treebank file into a text file."""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import myutils
import nltk
from mylogger import main_logger

logger = main_logger.getChild(__name__)


@dataclass
class Args:
    input_tree_file: str
    output_file: str


if __name__ == "__main__":
    # Set args.
    parser: argparse.ArgumentParser = myutils.gen_parser(Args, required=True)
    args: Args = Args(**myutils.parse_and_filter(parser, sys.argv[1:]))

    output_file: Path = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Start conversion!!!")
    sents: list[str] = []
    with Path(args.input_tree_file).open(mode="r") as f:
        for l in f:
            tree: nltk.Tree = nltk.Tree.fromstring(l, remove_empty_top_bracketing=True)

            sent: str = " ".join(tree.leaves())

            sents.append(sent)

    with output_file.open(mode="w") as g:
        g.write("\n".join(sents))

    logger.info(f"Loaded {len(sents)} sentences.")
    logger.info("Finish conversion!!!")
