import re

import nltk
import numpy as np


class NTInsertPosLimitError(Exception):
    pass


class GeneralizedActionDict:
    def __init__(
        self,
        nonterminals: list[str],
        nt_insert_pos_limit: int,
    ):
        assert isinstance(nonterminals, list)

        self.nonterminals = nonterminals
        self.nt_insert_pos_limit = nt_insert_pos_limit
        self.num_nts = len(nonterminals)

        # Do not save i2a to reduce unnecessary memory use.
        self.action_size: int = 0

        # Add <PAD>
        self.action_size += 1
        self.padding_idx = self.action_size - 1

        # Add REDUCE, SHIFT
        # Note that we do not use FINISH action. Instead, we finish parsing when EOS token is shifted.
        self.action_size += 2
        self.shift_idx = self.action_size - 2
        self.reduce_idx = self.action_size - 1

        # Add NT actions.
        # NT(X;n): insert "(X" in the left of n-th complete node.
        # NT(X;0) means inserting "(X" onto the top of stack.
        self.nt_begin_idx: int = self.action_size
        self.action_size += len(nonterminals) * (nt_insert_pos_limit + 1)
        self.nt_end_idx: int = self.action_size - 1
        self.num_actions_for_each_nt: int = self.nt_insert_pos_limit + 1

    def is_pad(self, a: int):
        return a == self.padding_idx

    def is_shift(self, a: int):
        return a == self.shift_idx

    def is_reduce(self, a: int):
        return a == self.reduce_idx

    def is_nt(self, a: int):
        return (self.nt_begin_idx <= a) and (a <= self.nt_end_idx)

    def nt_id(self, a: int):
        return (a - self.nt_begin_idx) // self.num_actions_for_each_nt

    def nt_pos(self, a: int):
        return (a - self.nt_begin_idx) % self.num_actions_for_each_nt

    def i2a(self, a: int) -> str:
        if self.is_pad(a):
            return "<PAD>"

        elif self.is_shift(a):
            return "SHIFT"

        elif self.is_reduce(a):
            return "REDUCE"

        elif self.is_nt(a):
            nt_id = self.nt_id(a)
            nt_pos = self.nt_pos(a)
            return f"NT({self.nonterminals[nt_id]};{nt_pos})"

        else:
            raise Exception(f"Invalid action id: {a}")

    def a2i(self, a: str) -> int:
        if a == "<PAD>":
            return self.padding_idx
        elif a == "SHIFT":
            return self.shift_idx
        elif a == "REDUCE":
            return self.reduce_idx

        elif a.startswith("NT("):
            m = re.findall(r"NT\((.+);(\d+)\)", a)
            assert len(m) == 1 and len(m[0]) == 2

            nt_id = self.nonterminals.index(
                m[0][0]
            )  # An exception may raised when passing an nt not in self.nonterminals.
            nt_pos = int(m[0][1])
            # Make sure the nt_pos is in the range of this action_dict.
            # assert nt_pos <= self.nt_insert_pos_limit
            if nt_pos > self.nt_insert_pos_limit:
                raise NTInsertPosLimitError

            a_id = self.nt_begin_idx + nt_id * self.num_actions_for_each_nt + nt_pos
            return a_id

        else:
            raise Exception(f"Invalid action: {a}")

    def to_id(self, actions: list[str]) -> list[int]:
        return [self.a2i(a) for a in actions]

    # NOT batched.
    def build_tree_str(self, action_ids: list[int], tokens: list[str]):
        token_count: int = 0
        stack: list[str] = []

        for a_id in action_ids:
            if self.is_nt(a_id):
                nt_pos = self.nt_pos(a_id)
                comp_node_count = 0

                for pos in range(len(stack), -1, -1):
                    if pos < len(stack) and not ("(" in stack[pos] and ")" not in stack[pos]):
                        comp_node_count += 1

                    if nt_pos == comp_node_count:
                        stack.insert(pos, f"({self.nonterminals[self.nt_id(a_id)]}")
                        break

            elif self.is_shift(a_id):
                stack.append(f"{tokens[token_count]}")
                token_count += 1

            elif self.is_reduce(a_id):
                open_idx = len(stack) - 1
                while not ("(" in stack[open_idx] and ")" not in stack[open_idx]):
                    # find until open elem (only '(' exists) is found
                    open_idx -= 1
                reduced = " ".join(stack[open_idx:] + [")"])
                stack = stack[:open_idx]

                stack.append(reduced)

        return " ".join(stack).replace(" )", ")")

    @staticmethod
    def get_topdown_actions(tree: nltk.Tree, pos_limit: int | None = None) -> list[str]:
        if pos_limit is not None and pos_limit < 0:
            raise NTInsertPosLimitError(f"top-down: nt_pos exceeds the pos_limit: {0} ({pos_limit})")

        actions_str: list[str] = []

        def calc_topdown_actions_sub(t: nltk.Tree):
            current_root: str = t.label()
            # NT insert position is always 0 for top-down strategy.
            actions_str.append(f"NT({current_root};{0})")

            for child in t:
                if isinstance(child, str):
                    actions_str.append("SHIFT")
                    # actions_str.append(f"GEN({child})")
                else:
                    # Otherwise the child is nltk.Tree.
                    calc_topdown_actions_sub(t=child)

            actions_str.append("REDUCE")

        calc_topdown_actions_sub(t=tree)

        # Shift eos token to finish.
        actions_str.append("SHIFT")

        return actions_str

    @staticmethod
    def get_bottomup_actions(tree: nltk.Tree, pos_limit: int | None = None) -> list[str]:
        actions_str: list[str] = []

        def calc_bottomup_actions_sub(t: nltk.Tree):
            current_root: str = t.label()

            for child in t:
                if isinstance(child, str):
                    actions_str.append("SHIFT")
                    # actions_str.append(f"GEN({child})")
                else:
                    # Otherwise the child is nltk.Tree.
                    calc_bottomup_actions_sub(t=child)

            # NT insert position is not constant for bottom-up strategy.

            nt_pos = len(t)
            if pos_limit is not None:
                if nt_pos > pos_limit:
                    raise NTInsertPosLimitError(f"bottom-up: nt_pos exceeds the pos_limit: {nt_pos} ({pos_limit})")

            actions_str.append(f"NT({current_root};{nt_pos})")
            actions_str.append("REDUCE")

        calc_bottomup_actions_sub(t=tree)

        # Shift eos token to finish.
        actions_str.append("SHIFT")

        return actions_str

    @staticmethod
    def get_left_n_corner_actions(tree: nltk.Tree, n: int, pos_limit: int | None = None) -> list[str]:
        actions_str: list[str] = []

        def calc_leftcorner_actions_sub(t: nltk.Tree):
            current_root: str = t.label()

            nt_opened: bool = False

            for i, child in enumerate(t):
                if isinstance(child, str):
                    actions_str.append("SHIFT")
                    # actions_str.append(f"GEN({child})")
                else:
                    # Otherwise the child is nltk.Tree.
                    calc_leftcorner_actions_sub(t=child)

                if i + 1 == n:
                    nt_pos = i + 1

                    if pos_limit is not None:
                        if nt_pos > pos_limit:
                            raise NTInsertPosLimitError(
                                f"left-n-corner: nt_pos exceeds the pos_limit: {nt_pos} ({pos_limit})"
                            )

                    actions_str.append(f"NT({current_root};{nt_pos})")

                    nt_opened = True

            # If n is larger than number of childs, then the prediction would be bottom-up.
            if not nt_opened:
                nt_pos = len(t)

                if pos_limit is not None:
                    if nt_pos > pos_limit:
                        raise NTInsertPosLimitError(
                            f"left-n-corner: nt_pos exceeds the pos_limit: {nt_pos} ({pos_limit})"
                        )

                actions_str.append(f"NT({current_root};{nt_pos})")

            actions_str.append("REDUCE")

        calc_leftcorner_actions_sub(t=tree)

        # Shift eos token to finish.
        actions_str.append("SHIFT")

        return actions_str

    @staticmethod
    def get_uniform_speculation_actions(tree: nltk.Tree, real_pos: float, pos_limit: int | None = None) -> list[str]:
        """Defines action sequences of uniform speculation strategy."""

        actions_str: list[str] = []

        assert 0.0 <= real_pos and real_pos < 1.0

        def calc_uniform_speculation_actions_sub(
            t: nltk.Tree,
        ):
            # Convert real-value position parameter to integer position parameter.
            nt_pos = int(np.floor(real_pos * (len(t) + 1)))

            if pos_limit is not None:
                if nt_pos > pos_limit:
                    raise NTInsertPosLimitError(
                        f"uniform-speculation: nt_pos exceeds the pos_limit: {nt_pos} ({pos_limit})"
                    )

            nt_label: str = t.label()

            for i, child in enumerate(t):
                # If i childs are completed, open parent nt.
                if nt_pos == i:
                    actions_str.append(f"NT({nt_label};{nt_pos})")

                # SHIFT the leaf.
                if isinstance(child, str):
                    actions_str.append("SHIFT")

                else:
                    # Otherwise, the child is nltk.Tree.
                    calc_uniform_speculation_actions_sub(
                        t=child,
                    )

            # Add bottom-up nt.
            if nt_pos == len(t):
                actions_str.append(f"NT({nt_label};{nt_pos})")

            # Close the parent nt.
            actions_str.append("REDUCE")

        calc_uniform_speculation_actions_sub(
            t=tree,
        )

        # Shift eos token to finish.
        actions_str.append("SHIFT")

        return actions_str

    @staticmethod
    def get_local_first_actions(tree: nltk.Tree, height: float, pos_limit: int | None = None) -> list[str]:
        """Nodes that have smaller height than the threshold (locacl structure nodes) are predicted top-down and otherwise bottom-up."""

        if pos_limit is not None and pos_limit < 0:
            raise NTInsertPosLimitError(f"top-down: nt_pos exceeds the pos_limit: {0} ({pos_limit})")

        actions_str: list[str] = []

        def calc_local_first_actions_sub(
            t: nltk.Tree,
        ):
            node_height: int = t.height() - 1  # We need to sbtract 1 to give height 0 to the leaves.

            if node_height <= height:
                topdown = True
            else:
                topdown = False  # This means the node is predicted bottom-up.

            nt_label: str = t.label()

            # Predict the node top-down.
            if topdown:
                actions_str.append(f"NT({nt_label};{0})")

            for child in t:
                # SHIFT the leaf.
                if isinstance(child, str):
                    actions_str.append("SHIFT")

                else:
                    # Otherwise, the child is nltk.Tree.
                    calc_local_first_actions_sub(
                        t=child,
                    )

            # Otherwise, predict the node bottom-up.
            if not topdown:
                nt_pos = len(t)
                if pos_limit is not None:
                    if nt_pos > pos_limit:
                        raise NTInsertPosLimitError(
                            f"local-first: nt_pos exceeds the pos_limit: {nt_pos} ({pos_limit})"
                        )

                actions_str.append(f"NT({nt_label};{nt_pos})")

            # Close the parent nt.
            actions_str.append("REDUCE")

        calc_local_first_actions_sub(
            t=tree,
        )

        # Shift eos token to finish.
        actions_str.append("SHIFT")

        return actions_str

    @staticmethod
    def get_global_first_actions(tree: nltk.Tree, depth: float, pos_limit: int | None = None) -> list[str]:
        """Nodes that have smaller depth than the threshold (global structure nodes) are predicted top-down and otherwise bottom-up."""

        if pos_limit is not None and pos_limit < 0:
            raise NTInsertPosLimitError(f"top-down: nt_pos exceeds the pos_limit: {0} ({pos_limit})")

        actions_str: list[str] = []

        def calc_global_first_actions_sub(
            t: nltk.Tree,
            cur_depth: int,
        ):
            if cur_depth <= depth:
                topdown = True
            else:
                topdown = False  # This means the node is predicted bottom-up.

            nt_label: str = t.label()

            # Predict the node top-down.
            if topdown:
                actions_str.append(f"NT({nt_label};{0})")

            for child in t:
                # SHIFT the leaf.
                if isinstance(child, str):
                    actions_str.append("SHIFT")

                else:
                    # Otherwise, the child is nltk.Tree.
                    calc_global_first_actions_sub(
                        t=child,
                        cur_depth=cur_depth + 1,
                    )

            # Otherwise, predict the node bottom-up.
            if not topdown:
                nt_pos = len(t)
                if pos_limit is not None:
                    if nt_pos > pos_limit:
                        raise NTInsertPosLimitError(
                            f"local-first: nt_pos exceeds the pos_limit: {nt_pos} ({pos_limit})"
                        )

                actions_str.append(f"NT({nt_label};{nt_pos})")

            # Close the parent nt.
            actions_str.append("REDUCE")

        calc_global_first_actions_sub(
            t=tree,
            cur_depth=0,
        )

        # Shift eos token to finish.
        actions_str.append("SHIFT")

        return actions_str
