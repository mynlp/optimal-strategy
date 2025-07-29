import json
import unittest
from pathlib import Path

import nltk
from action_dict import GeneralizedActionDict

# from vocab import SentencePieceVocabulary, Vocabulary


class TestModels(unittest.TestCase):
    def test_i2a_and_a2i(self):
        # The count is set so that the order of the tokens are the same as vocab_l.

        action_dict = GeneralizedActionDict(
            nonterminals=["S", "NP"],
            nt_insert_pos_limit=1,
        )

        i2a_l = [
            "<PAD>",
            "SHIFT",
            "REDUCE",
            "NT(S;0)",
            "NT(S;1)",
            "NT(NP;0)",
            "NT(NP;1)",
        ]
        i2a = {i: a for i, a in enumerate(i2a_l)}
        a2i = {a: i for i, a in enumerate(i2a_l)}

        i2a_made = {i: action_dict.i2a(i) for i in range(action_dict.action_size)}
        a2i_made = {a: action_dict.a2i(a) for a in i2a_l}

        self.assertEqual(i2a, i2a_made)
        self.assertEqual(a2i, a2i_made)

    def test_build_tree_str(self):
        tokens: list[str] = ["th", "e", "do", "g", "b", "ark", "s", "<EOS>"]

        action_dict = GeneralizedActionDict(
            nonterminals=["S", "NP", "VP", "PP"],
            nt_insert_pos_limit=10,
        )

        # print(f"{a2i=}")

        actions = [
            "NT(NP;0)",
            "SHIFT",
            "SHIFT",
            "SHIFT",
            "SHIFT",
            "REDUCE",
            "SHIFT",
            "NT(S;2)",
            "SHIFT",
            "SHIFT",
            "NT(VP;3)",
            "REDUCE",
            "REDUCE",
            "SHIFT",  # Shift EOS.
        ]
        action_ids = action_dict.to_id(actions=actions)

        tree_str = action_dict.build_tree_str(action_ids=action_ids, tokens=tokens)
        self.assertEqual(tree_str, "(S (NP th e do g) (VP b ark s)) <EOS>")

    def test_get_fixed_actions(self):
        # The input to nltk.Tree must be tree.
        trees_wo_eos: list[str] = [
            "(S (NP The red cat) (VP eats) (NP the fish (PP on the table)))",
            "(S (NP I) (VP am jumping (PP on the table)))",
        ]
        trees_w_eos: list[str] = [
            "(S (NP The red cat) (VP eats) (NP the fish (PP on the table))) <EOS>",
            "(S (NP I) (VP am jumping (PP on the table))) <EOS>",
        ]
        tokens: list[list[str]] = [
            ["The", "red", "cat", "eats", "the", "fish", "on", "the", "table", "<EOS>"],
            ["I", "am", "jumping", "on", "the", "table", "<EOS>"],
        ]

        action_dict = GeneralizedActionDict(
            nonterminals=["S", "NP", "VP", "PP"],
            nt_insert_pos_limit=10,
        )

        for i in range(len(trees_wo_eos)):
            tree: nltk.Tree = nltk.Tree.fromstring(trees_wo_eos[i], remove_empty_top_bracketing=True)

            # Top-down.
            topdown_actions: list[str] = GeneralizedActionDict.get_topdown_actions(tree=tree)
            topdown_tree_str = action_dict.build_tree_str(
                action_ids=action_dict.to_id(topdown_actions), tokens=tokens[i]
            )
            self.assertEqual(trees_w_eos[i], topdown_tree_str)

            # Bottom-up.
            bottomup_actions: list[str] = GeneralizedActionDict.get_bottomup_actions(tree=tree)
            bottomup_tree_str = action_dict.build_tree_str(
                action_ids=action_dict.to_id(bottomup_actions), tokens=tokens[i]
            )
            self.assertEqual(trees_w_eos[i], bottomup_tree_str)

            # Left-n-corner.
            for n in [1, 2, 3]:
                left_n_corner_actions: list[str] = GeneralizedActionDict.get_left_n_corner_actions(tree=tree, n=n)
                left_n_corner_tree_str = action_dict.build_tree_str(
                    action_ids=action_dict.to_id(left_n_corner_actions), tokens=tokens[i]
                )
                self.assertEqual(trees_w_eos[i], left_n_corner_tree_str)

            # Uniform-speculation.
            for real_pos in [0.26, 0.35, 0.65, 0.74]:
                uniform_speculation_actions: list[str] = GeneralizedActionDict.get_uniform_speculation_actions(
                    tree=tree, real_pos=real_pos
                )
                uniform_speculation_tree_str = action_dict.build_tree_str(
                    action_ids=action_dict.to_id(uniform_speculation_actions), tokens=tokens[i]
                )
                self.assertEqual(trees_w_eos[i], uniform_speculation_tree_str)

            # Local-first.
            for height in [1, 2, 3]:
                local_first_actions: list[str] = GeneralizedActionDict.get_local_first_actions(tree=tree, height=height)
                local_first_tree_str = action_dict.build_tree_str(
                    action_ids=action_dict.to_id(local_first_actions), tokens=tokens[i]
                )
                self.assertEqual(trees_w_eos[i], local_first_tree_str)

            # Global-first.
            for depth in [1, 2, 3]:
                global_first_actions: list[str] = GeneralizedActionDict.get_global_first_actions(tree=tree, depth=depth)
                global_first_tree_str = action_dict.build_tree_str(
                    action_ids=action_dict.to_id(global_first_actions), tokens=tokens[i]
                )
                self.assertEqual(trees_w_eos[i], global_first_tree_str)

    @unittest.skip("Run preprocess first and set the filename to run the test.")
    def test_action_dict_with_processed_dataset(self):
        file: Path = Path("../tmp/processed_dataset/ptb/ptb-train.json")

        data = []
        nonterminals = None
        nt_insert_pos_limit = None

        with file.open(mode="r") as f:
            for l in f:
                d = json.loads(l.strip())

                if d["key"] == "sentence":
                    data.append(d)
                elif d["key"] == "nonterminals":
                    nonterminals = d["value"]
                elif d["key"] == "args":
                    nt_insert_pos_limit = d["value"]["nt_insert_pos_limit"]

        assert nonterminals is not None
        assert nt_insert_pos_limit is not None

        action_dict = GeneralizedActionDict(
            nonterminals=nonterminals,
            nt_insert_pos_limit=nt_insert_pos_limit,
        )

        # Here, we use original tokens to recover <unk> tokens to compare with original trees.
        for sent_data in data:
            for strategy_key in sent_data["strategies"]:
                action_ids = sent_data["strategies"][strategy_key]["action_ids"]
                tree_str = action_dict.build_tree_str(
                    action_ids=action_ids,
                    tokens=sent_data["tokens_w_eos"],  # Use original tokens.
                )

            self.assertEqual(sent_data["tree_str_w_eos"], tree_str)


if __name__ == "__main__":
    unittest.main()
