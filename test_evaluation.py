import unittest

from utils import (
    calc_category_parsing_scores,
    calc_labeled_f1,
    calc_parsing_scores,
    calc_prec_recall_f1,
    get_unlabeled_spans,
    tree_to_span,
)


class TestModels(unittest.TestCase):
    def test_parsing_scores(self):
        gold_tree_str = "(S (NP (NP The red cat)) (VP eats) (NP the fish (PP on the table)))"
        pred_tree_str = "(S (NP (VP (VP The red cat))) (NP (VP eats) (S the) (NP fish (VP on the table))))"
        pred_tree_str_2 = (
            "The (NP (VP (VP red cat))) (NP (VP eats) the (NP fish (VP on the))) table"  # Non-tree example.
        )

        correct_gold_spans: dict[tuple[str, int, int], int] = {
            ("S", 0, 8): 1,
            ("NP", 0, 2): 2,
            ("VP", 3, 3): 1,
            ("NP", 4, 8): 1,
            ("PP", 6, 8): 1,
        }
        correct_pred_spans: dict[tuple[str, int, int], int] = {
            ("S", 0, 8): 1,
            ("NP", 0, 2): 1,
            ("VP", 0, 2): 2,
            ("NP", 3, 8): 1,
            ("VP", 3, 3): 1,
            ("S", 4, 4): 1,
            ("NP", 5, 8): 1,
            ("VP", 6, 8): 1,
        }
        correct_pred_spans_2: dict[tuple[str, int, int], int] = {
            ("NP", 1, 2): 1,
            ("VP", 1, 2): 2,
            ("NP", 3, 7): 1,
            ("VP", 3, 3): 1,
            ("NP", 5, 7): 1,
            ("VP", 6, 7): 1,
        }

        correct_unlabeled_gold_spans: dict[tuple[int, int], int] = {
            (0, 8): 1,
            (0, 2): 2,
            (3, 3): 1,
            (4, 8): 1,
            (6, 8): 1,
        }
        correct_unlabeled_pred_spans: dict[tuple[int, int], int] = {
            (0, 8): 1,
            (0, 2): 3,
            (3, 8): 1,
            (3, 3): 1,
            (4, 4): 1,
            (5, 8): 1,
            (6, 8): 1,
        }
        correct_unlabeled_pred_spans_2: dict[tuple[int, int], int] = {
            (1, 2): 3,
            (3, 7): 1,
            (3, 3): 1,
            (5, 7): 1,
            (6, 7): 1,
        }

        labeled_tp = 3
        labeled_fp = 6
        labeled_fn = 3
        unlabeled_tp = 5
        unlabeled_fp = 4
        unlabeled_fn = 1

        labeled_tp_2 = 1
        labeled_fp_2 = 6
        labeled_fn_2 = 5
        unlabeled_tp_2 = 1
        unlabeled_fp_2 = 6
        unlabeled_fn_2 = 5

        # tp, fp, fn for each category.
        cat_tp = {
            "S": 1,
            "NP": 1,
            "VP": 1,
            "PP": 0,
        }
        cat_fp = {
            "S": 1,
            "NP": 2,
            "VP": 3,
            "PP": 0,
        }
        cat_fn = {
            "S": 0,
            "NP": 2,
            "VP": 0,
            "PP": 1,
        }
        cat_scores = {cat: (cat_tp[cat], cat_fp[cat], cat_fn[cat]) for cat in cat_tp}

        cat_tp_2 = {
            "S": 0,
            "NP": 0,
            "VP": 1,
            "PP": 0,
        }
        cat_fp_2 = {
            "S": 0,
            "NP": 3,
            "VP": 3,
            "PP": 0,
        }
        cat_fn_2 = {
            "S": 1,
            "NP": 3,
            "VP": 0,
            "PP": 1,
        }
        cat_scores_2 = {cat: (cat_tp_2[cat], cat_fp_2[cat], cat_fn_2[cat]) for cat in cat_tp_2}

        assert sum(cat_tp.values()) == labeled_tp
        assert sum(cat_fp.values()) == labeled_fp
        assert sum(cat_fn.values()) == labeled_fn
        assert sum(cat_tp_2.values()) == labeled_tp_2
        assert sum(cat_fp_2.values()) == labeled_fp_2
        assert sum(cat_fn_2.values()) == labeled_fn_2

        labeled_prec = labeled_tp / (labeled_tp + labeled_fp)
        labeled_recall = labeled_tp / (labeled_tp + labeled_fn)
        labeled_f1 = 2 * labeled_prec * labeled_recall / (labeled_prec + labeled_recall)
        unlabeled_prec = unlabeled_tp / (unlabeled_tp + unlabeled_fp)
        unlabeled_recall = unlabeled_tp / (unlabeled_tp + unlabeled_fn)
        unlabeled_f1 = 2 * unlabeled_prec * unlabeled_recall / (unlabeled_prec + unlabeled_recall)

        labeled_prec_2 = labeled_tp_2 / (labeled_tp_2 + labeled_fp_2)
        labeled_recall_2 = labeled_tp_2 / (labeled_tp_2 + labeled_fn_2)
        labeled_f1_2 = 2 * labeled_prec_2 * labeled_recall_2 / (labeled_prec_2 + labeled_recall_2)
        unlabeled_prec_2 = unlabeled_tp_2 / (unlabeled_tp_2 + unlabeled_fp_2)
        unlabeled_recall_2 = unlabeled_tp_2 / (unlabeled_tp_2 + unlabeled_fn_2)
        unlabeled_f1_2 = 2 * unlabeled_prec_2 * unlabeled_recall_2 / (unlabeled_prec_2 + unlabeled_recall_2)

        gold_spans = tree_to_span(tree_str=gold_tree_str)
        pred_spans = tree_to_span(tree_str=pred_tree_str)
        pred_spans_2 = tree_to_span(tree_str=pred_tree_str_2)
        tmp_tp, tmp_fp, tmp_fn = calc_parsing_scores(pred_spans=pred_spans, gold_spans=gold_spans)
        tmp_tp_2, tmp_fp_2, tmp_fn_2 = calc_parsing_scores(pred_spans=pred_spans_2, gold_spans=gold_spans)

        self.assertEqual(gold_spans, correct_gold_spans)
        self.assertEqual(pred_spans, correct_pred_spans)
        self.assertEqual(pred_spans_2, correct_pred_spans_2)

        self.assertEqual((tmp_tp, tmp_fp, tmp_fn), (labeled_tp, labeled_fp, labeled_fn))
        self.assertEqual((tmp_tp_2, tmp_fp_2, tmp_fn_2), (labeled_tp_2, labeled_fp_2, labeled_fn_2))

        self.assertEqual(calc_category_parsing_scores(pred_spans=pred_spans, gold_spans=gold_spans), cat_scores)
        self.assertEqual(calc_category_parsing_scores(pred_spans=pred_spans_2, gold_spans=gold_spans), cat_scores_2)

        self.assertEqual(
            calc_prec_recall_f1(tp=tmp_tp, fp=tmp_fp, fn=tmp_fn),
            (labeled_prec, labeled_recall, labeled_f1),
        )
        self.assertEqual(calc_labeled_f1(pred_spans=pred_spans, gold_spans=gold_spans), labeled_f1)

        self.assertEqual(
            calc_prec_recall_f1(tp=tmp_tp_2, fp=tmp_fp_2, fn=tmp_fn_2),
            (labeled_prec_2, labeled_recall_2, labeled_f1_2),
        )
        self.assertEqual(
            calc_labeled_f1(pred_spans=pred_spans_2, gold_spans=gold_spans),
            labeled_f1_2,
        )

        unlabeled_gold_spans = get_unlabeled_spans(span_count=gold_spans)
        unlabeled_pred_spans = get_unlabeled_spans(span_count=pred_spans)
        unlabeled_pred_spans_2 = get_unlabeled_spans(span_count=pred_spans_2)

        self.assertEqual(unlabeled_gold_spans, correct_unlabeled_gold_spans)
        self.assertEqual(unlabeled_pred_spans, correct_unlabeled_pred_spans)
        self.assertEqual(unlabeled_pred_spans_2, correct_unlabeled_pred_spans_2)

        unlabeled_tmp_tp, unlabeled_tmp_fp, unlabeled_tmp_fn = calc_parsing_scores(
            pred_spans=unlabeled_pred_spans, gold_spans=unlabeled_gold_spans
        )
        unlabeled_tmp_tp_2, unlabeled_tmp_fp_2, unlabeled_tmp_fn_2 = calc_parsing_scores(
            pred_spans=unlabeled_pred_spans_2, gold_spans=unlabeled_gold_spans
        )

        self.assertEqual(
            (unlabeled_tmp_tp, unlabeled_tmp_fp, unlabeled_tmp_fn),
            (unlabeled_tp, unlabeled_fp, unlabeled_fn),
        )
        self.assertEqual(
            calc_prec_recall_f1(tp=unlabeled_tmp_tp, fp=unlabeled_tmp_fp, fn=unlabeled_tmp_fn),
            (unlabeled_prec, unlabeled_recall, unlabeled_f1),
        )
        self.assertEqual(
            (unlabeled_tmp_tp_2, unlabeled_tmp_fp_2, unlabeled_tmp_fn_2),
            (unlabeled_tp_2, unlabeled_fp_2, unlabeled_fn_2),
        )
        self.assertEqual(
            calc_prec_recall_f1(tp=unlabeled_tmp_tp_2, fp=unlabeled_tmp_fp_2, fn=unlabeled_tmp_fn_2),
            (unlabeled_prec_2, unlabeled_recall_2, unlabeled_f1_2),
        )


#    def test_calc_sent_nt_insert_pos(self):
#        max_comp_nodes = 20
#        action_dict = GeneralizedActionDict(nonterminals=["S", "NP", "VP", "PP"], nt_insert_pos_limit=max_comp_nodes)
#
#        tree_str = "(S (NP The red cat) (VP eats) (NP the fish (PP on the table)))"
#        tokens = ["The", "red", "cat", "eats", "the", "fish", "on", "the", "table"]
#        is_subword_end = [True for _ in range(len(tokens))]
#
#        topdown_action: list[str] = [
#            "NT(S;0)",
#            "NT(NP;0)",
#            "SHIFT",
#            "SHIFT",
#            "SHIFT",
#            "REDUCE",
#            "NT(VP;0)",
#            "SHIFT",
#            "REDUCE",
#            "NT(NP;0)",
#            "SHIFT",
#            "SHIFT",
#            "NT(PP;0)",
#            "SHIFT",
#            "SHIFT",
#            "SHIFT",
#            "REDUCE",
#            "REDUCE",
#            "REDUCE",
#        ]
#        topdown_action_ids = [action_dict.a2i(a) for a in topdown_action]
#        topdown_nt_insert_pos = {"0": 5}
#
#        bottomup_action: list[str] = [
#            "SHIFT",
#            "SHIFT",
#            "SHIFT",
#            "NT(NP;3)",
#            "REDUCE",
#            "SHIFT",
#            "NT(VP;1)",
#            "REDUCE",
#            "SHIFT",
#            "SHIFT",
#            "SHIFT",
#            "SHIFT",
#            "SHIFT",
#            "NT(PP;3)",
#            "REDUCE",
#            "NT(NP;3)",
#            "REDUCE",
#            "NT(S;3)",
#            "REDUCE",
#        ]
#        bottomup_action_ids = [action_dict.a2i(a) for a in bottomup_action]
#        bottomup_nt_insert_pos = {"inf": 5}
#
#        leftcorner_action: list[str] = [
#            "SHIFT",
#            "NT(NP;1)",
#            "SHIFT",
#            "SHIFT",
#            "REDUCE",
#            "NT(S;1)",
#            "SHIFT",
#            "NT(VP;1)",
#            "REDUCE",
#            "SHIFT",
#            "NT(NP;1)",
#            "SHIFT",
#            "SHIFT",
#            "NT(PP;1)",
#            "SHIFT",
#            "SHIFT",
#            "REDUCE",
#            "REDUCE",
#            "REDUCE",
#        ]
#        leftcorner_action_ids = [action_dict.a2i(a) for a in leftcorner_action]
#        leftcorner_nt_insert_pos = {"1": 4, "inf": 1}
#
#        mixed_action: list[str] = [
#            "NT(NP;0)",
#            "SHIFT",
#            "SHIFT",
#            "SHIFT",
#            "REDUCE",
#            "NT(VP;0)",
#            "NT(S;1)",
#            "SHIFT",
#            "REDUCE",
#            "SHIFT",
#            "SHIFT",
#            "SHIFT",
#            "SHIFT",
#            "NT(PP;2)",
#            "NT(NP;4)",
#            "SHIFT",
#            "REDUCE",
#            "REDUCE",
#            "REDUCE",
#        ]
#        mixed_action_ids = [action_dict.a2i(a) for a in mixed_action]
#        mixed_nt_insert_pos = {
#            "0": 2,
#            "1.5": 1,
#            "2": 1,
#            "2.5": 1,
#        }
#
#        mixed2_action: list[str] = [
#            "SHIFT",
#            "NT(S;1)",
#            "NT(NP;1)",
#            "SHIFT",
#            "SHIFT",
#            "REDUCE",
#            "SHIFT",
#            "NT(VP;1)",
#            "REDUCE",
#            "SHIFT",
#            "SHIFT",
#            "SHIFT",
#            "SHIFT",
#            "SHIFT",
#            "NT(NP;5)",
#            "NT(PP;3)",
#            "REDUCE",
#            "REDUCE",
#            "REDUCE",
#        ]
#        mixed2_action_ids = [action_dict.a2i(a) for a in mixed2_action]
#        mixed2_nt_insert_pos = {
#            "0.5": 1,
#            "1": 1,
#            "2.5": 1,
#            "inf": 2,
#        }
#
#        unary_tree_str: str = "(S (NP (VP (PP (NP hoge)))))"
#        unary_tokens: list[str] = ["hoge"]
#        unary_action: list[str] = [
#            "SHIFT",
#            "NT(S;1)",
#            "NT(NP;1)",
#            "NT(VP;1)",
#            "NT(PP;1)",
#            "NT(NP;1)",
#            "REDUCE",
#            "REDUCE",
#            "REDUCE",
#            "REDUCE",
#            "REDUCE",
#        ]
#        unary_action_ids = [action_dict.a2i(a) for a in unary_action]
#        unary_nt_insert_pos = {
#            "0.5": 4,
#            "inf": 1,
#        }
#
#        # First check whether the actions can reconstruct the original tree.
#        self.assertEqual(
#            tree_str,
#            action_dict.build_tree_str(action_ids=topdown_action_ids, tokens=tokens),
#        )
#        self.assertEqual(
#            tree_str,
#            action_dict.build_tree_str(action_ids=bottomup_action_ids, tokens=tokens),
#        )
#        self.assertEqual(
#            tree_str,
#            action_dict.build_tree_str(action_ids=leftcorner_action_ids, tokens=tokens),
#        )
#        self.assertEqual(
#            tree_str,
#            action_dict.build_tree_str(action_ids=mixed_action_ids, tokens=tokens),
#        )
#        self.assertEqual(
#            tree_str,
#            action_dict.build_tree_str(action_ids=mixed2_action_ids, tokens=tokens),
#        )
#        self.assertEqual(
#            unary_tree_str,
#            action_dict.build_tree_str(action_ids=unary_action_ids, tokens=unary_tokens),
#        )
#
#        # Next, check nt insert positions.
#        # Subword nt insert position should be the same as word in these cases.
#        # topdown
#        calc_topdown_nt_insert_pos, subw_calc_topdown_nt_insert_pos = calc_sent_nt_insert_pos(
#            is_subword_end=is_subword_end,
#            action_ids=topdown_action_ids,
#            action_dict=action_dict,
#        )
#
#        self.assertEqual(
#            topdown_nt_insert_pos,
#            calc_topdown_nt_insert_pos,
#        )
#        self.assertEqual(
#            topdown_nt_insert_pos,
#            subw_calc_topdown_nt_insert_pos,
#        )
#
#        # bottomup
#        calc_bottomup_nt_insert_pos, subw_calc_bottomup_nt_insert_pos = calc_sent_nt_insert_pos(
#            is_subword_end=is_subword_end,
#            action_ids=bottomup_action_ids,
#            action_dict=action_dict,
#        )
#
#        self.assertEqual(
#            bottomup_nt_insert_pos,
#            calc_bottomup_nt_insert_pos,
#        )
#        self.assertEqual(
#            bottomup_nt_insert_pos,
#            subw_calc_bottomup_nt_insert_pos,
#        )
#
#        # leftcorner
#        calc_leftcorner_nt_insert_pos, subw_calc_leftcorner_nt_insert_pos = calc_sent_nt_insert_pos(
#            is_subword_end=is_subword_end,
#            action_ids=leftcorner_action_ids,
#            action_dict=action_dict,
#        )
#
#        self.assertEqual(
#            leftcorner_nt_insert_pos,
#            calc_leftcorner_nt_insert_pos,
#        )
#        self.assertEqual(
#            leftcorner_nt_insert_pos,
#            subw_calc_leftcorner_nt_insert_pos,
#        )
#
#        # mixed
#        calc_mixed_nt_insert_pos, subw_calc_mixed_nt_insert_pos = calc_sent_nt_insert_pos(
#            is_subword_end=is_subword_end,
#            action_ids=mixed_action_ids,
#            action_dict=action_dict,
#        )
#
#        self.assertEqual(
#            mixed_nt_insert_pos,
#            calc_mixed_nt_insert_pos,
#        )
#        self.assertEqual(
#            mixed_nt_insert_pos,
#            subw_calc_mixed_nt_insert_pos,
#        )
#
#        # mixed2
#        calc_mixed2_nt_insert_pos, subw_calc_mixed2_nt_insert_pos = calc_sent_nt_insert_pos(
#            is_subword_end=is_subword_end,
#            action_ids=mixed2_action_ids,
#            action_dict=action_dict,
#        )
#
#        self.assertEqual(
#            mixed2_nt_insert_pos,
#            calc_mixed2_nt_insert_pos,
#        )
#        self.assertEqual(
#            mixed2_nt_insert_pos,
#            subw_calc_mixed2_nt_insert_pos,
#        )
#
#        # unary
#        calc_unary_nt_insert_pos, subw_calc_unary_nt_insert_pos = calc_sent_nt_insert_pos(
#            is_subword_end=is_subword_end,
#            action_ids=unary_action_ids,
#            action_dict=action_dict,
#        )
#
#        self.assertEqual(
#            unary_nt_insert_pos,
#            calc_unary_nt_insert_pos,
#        )
#        self.assertEqual(
#            unary_nt_insert_pos,
#            subw_calc_unary_nt_insert_pos,
#        )
#
#    def test_calc_sent_nt_insert_pos_subword(self):
#        max_comp_nodes = 20
#        action_dict = GeneralizedActionDict(nonterminals=["S", "NP", "VP", "PP"], nt_insert_pos_limit=max_comp_nodes)
#
#        # Note that left-corner-subw strategy cannot express this tree since a phrase must contain at least one word boundary for left-corner-subw.
#        tree_str = "(S (NP (NP T) he▁ (NP re d▁ c) a t▁) (VP eat s▁) (NP t (NP he▁ fish▁) (PP on▁ t he▁ t able▁)))"
#        tokens = [
#            "T",
#            "he▁",
#            "re",
#            "d▁",
#            "c",
#            "a",
#            "t▁",
#            "eat",
#            "s▁",
#            "t",
#            "he▁",
#            "fish▁",
#            "on▁",
#            "t",
#            "he▁",
#            "t",
#            "able▁",
#        ]
#        is_subword_end = ["▁" in token for token in tokens]
#
#        topdown_action: list[str] = [
#            "NT(S;0)",
#            "NT(NP;0)",
#            "NT(NP;0)",
#            "SHIFT",
#            "REDUCE",
#            "SHIFT",
#            "NT(NP;0)",
#            "SHIFT",
#            "SHIFT",
#            "SHIFT",
#            "REDUCE",
#            "SHIFT",
#            "SHIFT",
#            "REDUCE",
#            "NT(VP;0)",
#            "SHIFT",
#            "SHIFT",
#            "REDUCE",
#            "NT(NP;0)",
#            "SHIFT",
#            "NT(NP;0)",
#            "SHIFT",
#            "SHIFT",
#            "REDUCE",
#            "NT(PP;0)",
#            "SHIFT",
#            "SHIFT",
#            "SHIFT",
#            "SHIFT",
#            "SHIFT",
#            "REDUCE",
#            "REDUCE",
#            "REDUCE",
#        ]
#        topdown_action_ids = [action_dict.a2i(a) for a in topdown_action]
#        topdown_nt_insert_pos = {"0": 8}
#        subw_topdown_nt_insert_pos = {"0": 8}
#
#        bottomup_action: list[str] = [
#            "SHIFT",
#            "NT(NP;1)",
#            "REDUCE",
#            "SHIFT",
#            "SHIFT",
#            "SHIFT",
#            "SHIFT",
#            "NT(NP;3)",
#            "REDUCE",
#            "SHIFT",
#            "SHIFT",
#            "NT(NP;5)",
#            "REDUCE",
#            "SHIFT",
#            "SHIFT",
#            "NT(VP;2)",
#            "REDUCE",
#            "SHIFT",
#            "SHIFT",
#            "SHIFT",
#            "NT(NP;2)",
#            "REDUCE",
#            "SHIFT",
#            "SHIFT",
#            "SHIFT",
#            "SHIFT",
#            "SHIFT",
#            "NT(PP;5)",
#            "REDUCE",
#            "NT(NP;3)",
#            "REDUCE",
#            "NT(S;3)",
#            "REDUCE",
#        ]
#        bottomup_action_ids = [action_dict.a2i(a) for a in bottomup_action]
#        bottomup_nt_insert_pos = {"inf": 8}
#        subw_bottomup_nt_insert_pos = {"inf": 8}
#
#        # Not left-corner-subw
#        leftcorner_action: list[str] = [
#            "SHIFT",
#            "NT(NP;1)",
#            "REDUCE",
#            "NT(NP;1)",
#            "SHIFT",
#            "SHIFT",
#            "NT(NP;1)",
#            "SHIFT",
#            "SHIFT",
#            "REDUCE",
#            "SHIFT",
#            "SHIFT",
#            "REDUCE",
#            "NT(S;1)",
#            "SHIFT",
#            "NT(VP;1)",
#            "SHIFT",
#            "REDUCE",
#            "SHIFT",
#            "NT(NP;1)",
#            "SHIFT",
#            "NT(NP;1)",
#            "SHIFT",
#            "REDUCE",
#            "SHIFT",
#            "NT(PP;1)",
#            "SHIFT",
#            "SHIFT",
#            "SHIFT",
#            "SHIFT",
#            "REDUCE",
#            "REDUCE",
#            "REDUCE",
#        ]
#        leftcorner_action_ids = [action_dict.a2i(a) for a in leftcorner_action]
#        leftcorner_nt_insert_pos = {"1": 7, "inf": 1}
#        subw_leftcorner_nt_insert_pos = {"0.5": 3, "1": 4, "inf": 1}
#
#        mixed_action: list[str] = [
#            "NT(NP;0)",
#            "SHIFT",
#            "REDUCE",
#            "SHIFT",
#            "SHIFT",
#            "NT(NP;1)",
#            "SHIFT",
#            "SHIFT",
#            "REDUCE",
#            "SHIFT",
#            "NT(NP;4)",
#            "SHIFT",
#            "REDUCE",
#            "NT(VP;0)",
#            "NT(S;1)",
#            "SHIFT",
#            "SHIFT",
#            "REDUCE",
#            "SHIFT",
#            "SHIFT",
#            "NT(NP;2)",
#            "SHIFT",
#            "NT(NP;2)",
#            "REDUCE",
#            "SHIFT",
#            "SHIFT",
#            "SHIFT",
#            "SHIFT",
#            "NT(PP;4)",
#            "SHIFT",
#            "REDUCE",
#            "REDUCE",
#            "REDUCE",
#        ]
#        mixed_action_ids = [action_dict.a2i(a) for a in mixed_action]
#        mixed_nt_insert_pos = {
#            "0": 2,
#            "1": 1,
#            "1.5": 2,
#            "4": 2,
#            "inf": 1,
#        }
#        subw_mixed_nt_insert_pos = {
#            "0": 2,
#            # "0.5": 1,
#            # "1.5": 2,
#            # We do not count when a child is an unfinished subword.
#            "0.5": 2,
#            "1.5": 1,
#            "2.5": 1,
#            "3.5": 1,
#            "inf": 1,
#        }
#
#        mixed2_action: list[str] = [
#            "SHIFT",
#            "NT(S;1)",
#            "NT(NP;1)",
#            "NT(NP;1)",
#            "REDUCE",
#            "SHIFT",
#            "SHIFT",
#            "SHIFT",
#            "NT(NP;2)",
#            "SHIFT",
#            "REDUCE",
#            "SHIFT",
#            "SHIFT",
#            "REDUCE",
#            "SHIFT",
#            "NT(VP;1)",
#            "SHIFT",
#            "REDUCE",
#            "SHIFT",
#            "SHIFT",
#            "SHIFT",
#            "NT(NP;3)",
#            "NT(NP;2)",
#            "REDUCE",
#            "SHIFT",
#            "SHIFT",
#            "SHIFT",
#            "NT(PP;3)",
#            "SHIFT",
#            "SHIFT",
#            "REDUCE",
#            "REDUCE",
#            "REDUCE",
#        ]
#        mixed2_action_ids = [action_dict.a2i(a) for a in mixed2_action]
#        mixed2_nt_insert_pos = {
#            "0.5": 2,
#            "inf": 2,
#            "2": 1,
#            "1": 1,
#            "1.5": 1,
#            "3": 1,
#        }
#        subw_mixed2_nt_insert_pos = {
#            "0.5": 4,
#            "1": 1,
#            "2": 1,
#            "inf": 2,
#            # "1": 1,
#            # "1.5": 1,
#        }
#
#        # Left-corner-subw can only handle trees that have at least one subword_end in each phrase.
#        tree_str_leftcornersubw = (
#            "(S (NP T he▁ (NP re d▁ c) a t▁) (VP eat s▁) t (NP (NP he▁ fish▁) (PP on▁ t he▁ t able▁)))"
#        )
#        leftcornersubw_action: list[str] = [
#            "SHIFT",
#            "SHIFT",
#            "NT(NP;2)",
#            "SHIFT",
#            "SHIFT",
#            "NT(NP;2)",
#            "SHIFT",
#            "REDUCE",
#            "SHIFT",
#            "SHIFT",
#            "REDUCE",
#            "NT(S;1)",
#            "SHIFT",
#            "SHIFT",
#            "NT(VP;2)",
#            "REDUCE",
#            "SHIFT",
#            "SHIFT",
#            "NT(NP;1)",
#            "SHIFT",
#            "REDUCE",
#            "NT(NP;1)",
#            "SHIFT",
#            "NT(PP;1)",
#            "SHIFT",
#            "SHIFT",
#            "SHIFT",
#            "SHIFT",
#            "REDUCE",
#            "REDUCE",
#            "REDUCE",
#        ]
#        leftcornersubw_action_ids = [action_dict.a2i(a) for a in leftcornersubw_action]
#        leftcornersubw_nt_insert_pos = {
#            "2": 2,
#            "1": 4,
#            "inf": 1,
#        }
#        subw_leftcornersubw_nt_insert_pos = {"1": 6, "inf": 1}
#
#        # First check whether the actions can reconstruct the original tree.
#        self.assertEqual(
#            tree_str,
#            action_dict.build_tree_str(action_ids=topdown_action_ids, tokens=tokens),
#        )
#        self.assertEqual(
#            tree_str,
#            action_dict.build_tree_str(action_ids=bottomup_action_ids, tokens=tokens),
#        )
#        self.assertEqual(
#            tree_str,
#            action_dict.build_tree_str(action_ids=leftcorner_action_ids, tokens=tokens),
#        )
#        self.assertEqual(
#            tree_str,
#            action_dict.build_tree_str(action_ids=mixed_action_ids, tokens=tokens),
#        )
#        self.assertEqual(
#            tree_str,
#            action_dict.build_tree_str(action_ids=mixed2_action_ids, tokens=tokens),
#        )
#        self.assertEqual(
#            tree_str_leftcornersubw,
#            action_dict.build_tree_str(action_ids=leftcornersubw_action_ids, tokens=tokens),
#        )
#
#        # Next, check nt insert positions.
#        # Subword nt insert position should be the same as word in these cases.
#        # topdown
#        calc_topdown_nt_insert_pos, subw_calc_topdown_nt_insert_pos = calc_sent_nt_insert_pos(
#            is_subword_end=is_subword_end,
#            action_ids=topdown_action_ids,
#            action_dict=action_dict,
#        )
#
#        self.assertEqual(
#            topdown_nt_insert_pos,
#            calc_topdown_nt_insert_pos,
#        )
#        self.assertEqual(
#            subw_topdown_nt_insert_pos,
#            subw_calc_topdown_nt_insert_pos,
#        )
#
#        # bottomup
#        calc_bottomup_nt_insert_pos, subw_calc_bottomup_nt_insert_pos = calc_sent_nt_insert_pos(
#            is_subword_end=is_subword_end,
#            action_ids=bottomup_action_ids,
#            action_dict=action_dict,
#        )
#
#        self.assertEqual(
#            bottomup_nt_insert_pos,
#            calc_bottomup_nt_insert_pos,
#        )
#        self.assertEqual(
#            subw_bottomup_nt_insert_pos,
#            subw_calc_bottomup_nt_insert_pos,
#        )
#
#        # leftcorner
#        calc_leftcorner_nt_insert_pos, subw_calc_leftcorner_nt_insert_pos = calc_sent_nt_insert_pos(
#            is_subword_end=is_subword_end,
#            action_ids=leftcorner_action_ids,
#            action_dict=action_dict,
#        )
#
#        self.assertEqual(
#            leftcorner_nt_insert_pos,
#            calc_leftcorner_nt_insert_pos,
#        )
#        self.assertEqual(
#            subw_leftcorner_nt_insert_pos,
#            subw_calc_leftcorner_nt_insert_pos,
#        )
#
#        # mixed
#        calc_mixed_nt_insert_pos, subw_calc_mixed_nt_insert_pos = calc_sent_nt_insert_pos(
#            is_subword_end=is_subword_end,
#            action_ids=mixed_action_ids,
#            action_dict=action_dict,
#        )
#
#        self.assertEqual(
#            mixed_nt_insert_pos,
#            calc_mixed_nt_insert_pos,
#        )
#        self.assertEqual(
#            subw_mixed_nt_insert_pos,
#            subw_calc_mixed_nt_insert_pos,
#        )
#
#        # mixed2
#        calc_mixed2_nt_insert_pos, subw_calc_mixed2_nt_insert_pos = calc_sent_nt_insert_pos(
#            is_subword_end=is_subword_end,
#            action_ids=mixed2_action_ids,
#            action_dict=action_dict,
#        )
#
#        self.assertEqual(
#            mixed2_nt_insert_pos,
#            calc_mixed2_nt_insert_pos,
#        )
#        self.assertEqual(
#            subw_mixed2_nt_insert_pos,
#            subw_calc_mixed2_nt_insert_pos,
#        )
#
#        # leftcornersubw
#        calc_leftcornersubw_nt_insert_pos, subw_calc_leftcornersubw_nt_insert_pos = calc_sent_nt_insert_pos(
#            is_subword_end=is_subword_end,
#            action_ids=leftcornersubw_action_ids,
#            action_dict=action_dict,
#        )
#
#        self.assertEqual(
#            leftcornersubw_nt_insert_pos,
#            calc_leftcornersubw_nt_insert_pos,
#        )
#        self.assertEqual(
#            subw_leftcornersubw_nt_insert_pos,
#            subw_calc_leftcornersubw_nt_insert_pos,
#        )


if __name__ == "__main__":
    unittest.main()
