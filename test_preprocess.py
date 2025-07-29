import sys
import unittest

import nltk
from get_treebank import remove_preterminals
from utils import split_leaves_to_subwords


class TestPreprocess(unittest.TestCase):
    def test_remove_preterminals(self):
        tree_str = "(S (NP (DT The) (JJ new) (NN rate)) (VP (MD will) (VP (VB be) (ADJP (JJ payable) (NP (NNP Feb.) (CD 15))))) (. .))"
        goal_tree_str = "(S (NP The new rate) (VP will (VP be (ADJP payable (NP Feb. 15)))) .)"

        tree = nltk.Tree.fromstring(tree_str, remove_empty_top_bracketing=True)
        processed_tree = remove_preterminals(t=tree)
        processed_tree_str = processed_tree.pformat(margin=sys.maxsize)

        self.assertEqual(goal_tree_str, processed_tree_str)

    def test_split_leaves_to_subwords(self):
        tree_str = "(S (NP the dog) (VP barks))"
        tree = nltk.Tree.fromstring(tree_str, remove_empty_top_bracketing=True)

        # Note that '▁' here is not the same as '_'.
        subword_tree_str = "(S (NP th e▁ do g▁) (VP b ark s ▁))"

        pieces: list[str] = ["th", "e▁", "do", "g▁", "b", "ark", "s", "▁"]

        end_idxs = [i + 1 for i, p in enumerate(pieces) if "▁" in p]

        begin_idxs = [0] + end_idxs[:-1]
        subword_spans = list(zip(begin_idxs, end_idxs))  # map from original token idx to piece span idxs.

        transformed_tree, _ = split_leaves_to_subwords(
            t=tree, subword_spans=subword_spans, pieces=pieces, current_word_id=0
        )
        transformed_tree_str = transformed_tree.pformat(margin=sys.maxsize)

        self.assertEqual(subword_tree_str, transformed_tree_str)


if __name__ == "__main__":
    unittest.main()
