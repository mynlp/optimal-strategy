from venv import logger

import torch
import torch.nn.functional as F
from action_dict import GeneralizedActionDict
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils import *


class NoValidNextActionError(Exception):
    pass


class StackIndexError(Exception):
    pass


class MultiLayerLSTMCell(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        bias: bool = False,
        dropout: float = 0.0,
        layernorm: bool = True,
    ):
        super(MultiLayerLSTMCell, self).__init__()
        self.lstm = nn.ModuleList()
        self.lstm.append(nn.LSTMCell(input_size, hidden_size))
        for i in range(num_layers - 1):
            self.lstm.append(nn.LSTMCell(hidden_size, hidden_size))

        self.regularizer = nn.ModuleList()
        if layernorm:
            # Set layernorm as regularizer.
            for _ in range(num_layers):
                self.regularizer.append(nn.LayerNorm(normalized_shape=hidden_size))

        else:
            # Set dropout as regularizer.
            dropout_layer = nn.Dropout(dropout, inplace=True)
            for _ in range(num_layers):
                self.regularizer.append(dropout_layer)

        self.num_layers = num_layers

    def forward(self, input, prev):
        """

        :param input: (batch_size, input_size)
        :param prev: tuple of (h0, c0), each has size (batch, hidden_size, num_layers)
        """

        next_hidden = []
        next_cell = []

        if prev is None:
            prev = (
                input.new(input.size(0), self.lstm[0].hidden_size, self.num_layers).fill_(0),
                input.new(input.size(0), self.lstm[0].hidden_size, self.num_layers).fill_(0),
            )

        for i in range(self.num_layers):
            prev_hidden_i = prev[0][:, :, i]
            prev_cell_i = prev[1][:, :, i]
            if i == 0:
                next_hidden_i, next_cell_i = self.lstm[i](input, (prev_hidden_i, prev_cell_i))
            else:
                # Only apply regularizer to hidden state (but not the cell state)
                input_im1 = self.regularizer[i](input_im1)
                next_hidden_i, next_cell_i = self.lstm[i](input_im1, (prev_hidden_i, prev_cell_i))
            next_hidden += [next_hidden_i]
            next_cell += [next_cell_i]
            input_im1 = next_hidden_i

        next_hidden = torch.stack(next_hidden).permute(1, 2, 0)
        next_cell = torch.stack(next_cell).permute(1, 2, 0)
        return next_hidden, next_cell


class LSTMComposition(nn.Module):
    def __init__(self, dim: int, dropout: float, layernorm: bool):
        super(LSTMComposition, self).__init__()
        self.dim = dim
        self.rnn = nn.LSTM(dim, dim, bidirectional=True, batch_first=True)

        if layernorm:
            self.output = nn.Sequential(
                nn.LayerNorm(normalized_shape=dim * 2),
                nn.Linear(dim * 2, dim),
                # nn.LayerNorm(normalized_shape=dim),
                nn.ReLU(),
            )

        else:
            self.output = nn.Sequential(nn.Dropout(dropout, inplace=True), nn.Linear(dim * 2, dim), nn.ReLU())

        # memo: not sure why this is necessary.
        self.cache_size: int = 10000
        self.batch_index = torch.arange(0, self.cache_size, dtype=torch.long)  # cache with sufficient number.

    def forward(self, children, ch_lengths, nt, nt_id, stack_state):
        """

        :param children: (batch_size, max_num_children, input_dim)
        :param ch_lengths: (batch_size)
        :param nt: (batch_size, input_dim)
        :param nt_id: (batch_size)
        """
        lengths = ch_lengths + 2
        nt = nt.unsqueeze(1)
        elems = torch.cat([nt, children, torch.zeros_like(nt)], dim=1)
        # elems[self.batch_index[: elems.size(0)], lengths - 1] = nt.squeeze(1)
        if elems.size(0) < self.cache_size:
            elems[self.batch_index[: elems.size(0)], lengths - 1] = nt.squeeze(1)
        else:
            elems[torch.arange(0, elems.size(0), dtype=torch.long, device=children.device), lengths - 1] = nt.squeeze(1)

        packed = pack_padded_sequence(elems, lengths.int().cpu(), batch_first=True, enforce_sorted=False)
        h, _ = self.rnn(packed)
        h, _ = pad_packed_sequence(h, batch_first=True)

        gather_idx = (lengths - 2).unsqueeze(1).expand(-1, h.size(-1)).unsqueeze(1)
        fwd = h.gather(1, gather_idx).squeeze(1)[:, : self.dim]
        bwd = h[:, 1, self.dim :]
        c = torch.cat([fwd, bwd], dim=1)

        return self.output(c), None, None


class AttentionComposition(nn.Module):
    def __init__(self, w_dim: int, dropout: float, layernorm: bool, num_labels: int = 10):
        super(AttentionComposition, self).__init__()
        self.w_dim = w_dim
        self.num_labels = num_labels

        # Set regularizer:
        if layernorm:
            self.nt_emb2_regularizer = nn.LayerNorm(normalized_shape=w_dim * 2)
            self.weighted_child_regularizer = nn.LayerNorm(normalized_shape=w_dim * 2)

        else:
            dropout_layer = nn.Dropout(dropout, inplace=True)
            self.nt_emb2_regularizer = dropout_layer
            self.weighted_child_regularizer = dropout_layer

        self.rnn = nn.LSTM(w_dim, w_dim, bidirectional=True, batch_first=True)

        self.V = nn.Linear(2 * w_dim, 2 * w_dim, bias=False)
        self.nt_emb = nn.Embedding(num_labels, w_dim)  # o_nt in the Kuncoro et al. (2017)
        self.nt_emb2 = nn.Sequential(
            nn.Embedding(num_labels, w_dim * 2), self.nt_emb2_regularizer
        )  # t_nt in the Kuncoro et al. (2017)
        self.gate = nn.Sequential(nn.Linear(w_dim * 4, w_dim * 2), nn.Sigmoid())
        self.output = nn.Sequential(nn.Linear(w_dim * 2, w_dim), nn.ReLU())

    def forward(self, children, ch_lengths, nt, nt_id, stack_state):  # children: (batch_size, n_children, w_dim)
        packed = pack_padded_sequence(children, ch_lengths.int().cpu(), batch_first=True, enforce_sorted=False)
        h, _ = self.rnn(packed)
        h, _ = pad_packed_sequence(h, batch_first=True)  # (batch, n_children, 2*w_dim)

        rhs = torch.cat([self.nt_emb(nt_id), stack_state], dim=1)  # (batch_size, w_dim*2, 1)
        logit = (self.V(h) * rhs.unsqueeze(1)).sum(-1)  # equivalent to bmm(self.V(h), rhs.unsqueeze(-1)).squeeze(-1)
        len_mask = (
            ch_lengths.new_zeros(children.size(0), 1) + torch.arange(children.size(1), device=children.device)
        ) >= ch_lengths.unsqueeze(1)
        logit[len_mask] = -float("inf")

        attn = F.softmax(logit, -1)
        weighted_child = (h * attn.unsqueeze(-1)).sum(1)
        weighted_child = self.weighted_child_regularizer(weighted_child)

        nt2 = self.nt_emb2(nt_id)  # (batch_size, w_dim)
        gate_input = torch.cat([nt2, weighted_child], dim=-1)
        g = self.gate(gate_input)  # (batch_size, w_dim)
        c = g * nt2 + (1 - g) * weighted_child  # (batch_size, w_dim)

        return self.output(c), attn, g


class GeneralizedActionFixedStack:
    def __init__(
        self,
        initial_hidden: tuple[torch.Tensor, torch.Tensor],
        stack_size: int,
        input_size: int,
        batch_size: int,
        beam_size: int,
        sample_size: int,
    ):
        """
        initial_hidden: pair of next_hidden and next_cell of size [(batch_size, hidden_size, layer), (batch_size, hidden_size, layer)]
        """
        super(GeneralizedActionFixedStack, self).__init__()
        device = initial_hidden[1].device
        hidden_size = initial_hidden[0].size(-2)
        num_layers = initial_hidden[0].size(-1)

        assert initial_hidden[0].size(0) == batch_size

        parallel_size = (batch_size, sample_size, beam_size)
        self.batch_index = (
            (
                torch.arange(0, batch_size, dtype=torch.long, device=device)
                .unsqueeze(1)
                .expand(-1, sample_size * beam_size)
                .reshape(-1)
            ),
            torch.cat(
                [
                    torch.arange(0, sample_size, dtype=torch.long, device=device)
                    .unsqueeze(1)
                    .expand(-1, beam_size)
                    .reshape(-1)
                    for _ in range(batch_size)
                ]
            ),
            torch.cat(
                [torch.arange(0, beam_size, dtype=torch.long, device=device) for _ in range(batch_size * sample_size)]
            ),
        )

        # self.batch_size = initial_hidden[0].size(0)
        self.batch_size = batch_size
        self.beam_size = beam_size
        self.sample_size = sample_size
        self.stack_size = stack_size
        self.input_size = input_size
        # self.max_comp_nodes = max_comp_nodes
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.pointer = torch.zeros(parallel_size, dtype=torch.long, device=device)  # word pointer
        self.top_position = torch.zeros(parallel_size, dtype=torch.long, device=device)  # stack top position
        self.hiddens = initial_hidden[0].new_zeros(
            parallel_size + (stack_size + 1, hidden_size, num_layers), device=device
        )
        self.cells = initial_hidden[0].new_zeros(
            parallel_size + (stack_size + 1, hidden_size, num_layers), device=device
        )
        self.trees = initial_hidden[0].new_zeros(parallel_size + (stack_size, input_size), device=device)

        # self.hiddens[:, :, 0, 0] = initial_hidden[0]
        # self.cells[:, :, 0, 0] = initial_hidden[1]
        self.hiddens[:, :, 0, 0] = initial_hidden[0].unsqueeze(1)  # Need to add sample dimension.
        self.cells[:, :, 0, 0] = initial_hidden[1].unsqueeze(1)  # Need to add sample dimension.

        self.nt_index = torch.zeros(parallel_size + (stack_size,), dtype=torch.long, device=device)
        self.nt_ids = torch.zeros(parallel_size + (stack_size,), dtype=torch.long, device=device)
        self.nt_index_pos = (
            torch.tensor([-1], dtype=torch.long, device=device).expand(parallel_size).clone()
        )  # default is -1 (0 means zero-dim exists)

        self.attrs = [
            "pointer",
            "top_position",
            "hiddens",
            "cells",
            "trees",
            "nt_index",
            "nt_ids",
            "nt_index_pos",
        ]

    # TODO: deal with beam width for efficiency...?
    def hidden_head(self, offset: int = 0, batches: tuple[torch.Tensor, ...] | None = None) -> torch.Tensor:
        assert offset >= 0
        if batches is None:
            return self.hiddens[
                self.batch_index + (self.top_position.view(-1) - offset,)
            ]  # (batches, hidden_size, num_layers)
        else:
            return self.hiddens[batches + (self.top_position[batches] - offset,)]  # (batches, hidden_size, num_layers)

    def cell_head(self, offset: int = 0, batches: tuple[torch.Tensor, ...] | None = None) -> torch.Tensor:
        assert offset >= 0
        if batches is None:
            return self.cells[self.batch_index + ((self.top_position.view(-1) - offset),)]
        else:
            return self.cells[batches + (self.top_position[batches] - offset,)]

    def do_shift(
        self,
        shift_batches: tuple[torch.Tensor, ...],  # (batch, sample, beam)
        shifted_embs: torch.Tensor,  # (batches, input_size)
    ):
        # First check if stack size is enough.
        if (self.top_position[shift_batches] >= self.stack_size).any():
            # stack.top_position is equall to the number of elements on the stack.
            logger.warning("Stack is already full!!!! Cannot SHIFT anymore!!!!")
            raise StackIndexError

        assert shifted_embs.size() == (shift_batches[0].size(0), self.input_size)
        # Update stack-like structures.
        # top_position here is the position to be inserted.
        self.trees[shift_batches + (self.top_position[shift_batches],)] = shifted_embs

        self.pointer[shift_batches] = self.pointer[shift_batches] + 1
        # top_position here is the position to be inserted next.
        self.top_position[shift_batches] = self.top_position[shift_batches] + 1

    def do_nt(
        self,
        nt_batches: tuple[torch.Tensor, ...],
        nt_embs: torch.Tensor,
        nt_ids: torch.Tensor,
        nt_pos: torch.Tensor,
    ):
        # First check if stack size is enough.
        if (self.top_position[nt_batches] >= self.stack_size).any():
            # stack.top_position is equall to the number of elements on the stack.
            logger.warning("Stack is already full!!!! Cannot NT anymore!!!!")
            print(f"{nt_batches=}")
            raise StackIndexError

        # Update stack-like structures.
        # Update stack tree.

        # Note the offset of stack.top_position; top_position is equall to the number of elements on the stack.
        insert_nt_idx = self.top_position[nt_batches] - nt_pos

        assert insert_nt_idx.size() == (nt_batches[0].size(0),)
        num_elems_to_move = nt_pos
        max_num_elems = num_elems_to_move.max().item()

        elem_idx_order = (
            torch.arange(max_num_elems, device=insert_nt_idx.device)
            .unsqueeze(0)
            # .repeat(insert_nt_idx.size(0), 1)
            .expand(
                insert_nt_idx.size(0), -1
            )  # Since elem_idx_order is only used for reference (i.e., not written to), expand can be used (without cloning) instead of repeat to avoid memory copying.
        )  # (nt_batch_size, max_num_elems)
        assert elem_idx_order.size() == (nt_batches[0].size(0), max_num_elems)

        nt_batches_for_move = convert_to_advanced_index(
            batch_index=nt_batches,
            mask=elem_idx_order < num_elems_to_move.unsqueeze(-1),
            start_idx=insert_nt_idx,
        )

        nt_batches_for_move_tgt = nt_batches_for_move[:-1] + (nt_batches_for_move[-1] + 1,)
        # First, shift.
        self.trees[nt_batches_for_move_tgt] = self.trees[nt_batches_for_move]
        # Then, insert.
        self.trees[nt_batches + (insert_nt_idx,)] = nt_embs

        self.nt_index_pos[nt_batches] = self.nt_index_pos[nt_batches] + 1
        self.nt_ids[nt_batches + (self.nt_index_pos[nt_batches],)] = nt_ids
        self.top_position[nt_batches] = self.top_position[nt_batches] + 1
        self.nt_index[nt_batches + (self.nt_index_pos[nt_batches],)] = insert_nt_idx + 1

    def do_reduce(self, reduce_batches, new_child):
        # Update stack-like structures.
        prev_nt_position = self.nt_index[reduce_batches + (self.nt_index_pos[reduce_batches],)]
        child_length = self.top_position[reduce_batches] - prev_nt_position

        self.trees[reduce_batches + (prev_nt_position - 1,)] = new_child

        # Update pointers/positions.
        # +1 is for the reduced nt.
        self.nt_index_pos[reduce_batches] = self.nt_index_pos[reduce_batches] - 1
        self.top_position[reduce_batches] = prev_nt_position

    def collect_reduced_children(self, reduce_batches):
        """

        :param reduce_batches: Tuple of idx tensors (output of non_zero()).
        """
        nt_index_pos = self.nt_index_pos[reduce_batches]
        prev_nt_position = self.nt_index[reduce_batches + (nt_index_pos,)]
        reduced_nt_ids = self.nt_ids[reduce_batches + (nt_index_pos,)]
        reduced_nts = self.trees[reduce_batches + (prev_nt_position - 1,)]
        child_length = self.top_position[reduce_batches] - prev_nt_position
        max_ch_length = child_length.max()

        child_idx = prev_nt_position.unsqueeze(1) + torch.arange(max_ch_length, device=prev_nt_position.device)
        child_idx[child_idx >= self.stack_size] = (
            self.stack_size - 1
        )  # ceiled at maximum stack size (exceeding this may occur for some batches, but those should be ignored safely.)
        child_idx = child_idx.unsqueeze(-1).expand(
            -1, -1, self.trees.size(-1)
        )  # (num_reduced_batch, max_num_child, input_dim)
        reduced_children = torch.gather(self.trees[reduce_batches], 1, child_idx)
        return reduced_children, child_length, reduced_nts, reduced_nt_ids

    def update_hidden(self, new_hidden, new_cell, no_nop_batches):
        # debug
        # print(f"{self.hiddens.size()=}")
        # print(f"{no_nop_batches=}")
        # print(f"{new_hidden.size()=}")
        # debug

        # Do nothing for nop actions (e.g., PAD).
        self.hiddens[no_nop_batches + (self.top_position[no_nop_batches],)] = new_hidden
        self.cells[no_nop_batches + (self.top_position[no_nop_batches],)] = new_cell

    def move_beams(self, self_move_idxs, other: "GeneralizedActionFixedStack", move_idxs):
        self.pointer[self_move_idxs] = other.pointer[move_idxs]
        self.top_position[self_move_idxs] = other.top_position[move_idxs]
        self.hiddens[self_move_idxs] = other.hiddens[move_idxs]
        self.cells[self_move_idxs] = other.cells[move_idxs]
        self.trees[self_move_idxs] = other.trees[move_idxs]
        self.nt_index[self_move_idxs] = other.nt_index[move_idxs]
        self.nt_ids[self_move_idxs] = other.nt_ids[move_idxs]
        self.nt_index_pos[self_move_idxs] = other.nt_index_pos[move_idxs]


class ActionPath:
    def __init__(
        self,
        batch_size: int,
        beam_size: int,
        sample_size: int,
        max_actions: torch.Tensor,  # (batch_size,) maximum number of actions allowed for each batch.
        padding_idx: int,
        device: str,
    ):
        super(ActionPath, self).__init__()

        self.padding_idx = padding_idx

        parallel_size = (batch_size, sample_size, beam_size)

        max_actions_max = max_actions.max()

        self.max_actions = max_actions  # (batch_size,)

        self.actions = torch.full(
            parallel_size
            + (
                max_actions_max + 1,
            ),  # +1 is for the first <PAD> action to deal with existence of prev_actions for the first time step.
            padding_idx,
            dtype=torch.long,
            device=device,
        )
        self.actions_pos = self.actions.new_zeros(parallel_size)
        self.attrs = ["actions", "actions_pos"]

    def move_beams(self, self_idxs, source, source_idxs):
        self.actions[self_idxs] = source.actions[source_idxs]
        self.actions_pos[self_idxs] = source.actions_pos[source_idxs]

    def add(self, actions, active_idxs):
        # action_pos should be updated before actions to correctly hold prev_actions.
        self.actions_pos[active_idxs] += 1
        self.actions[active_idxs + (self.actions_pos[active_idxs],)] = actions[active_idxs]


class BeamItems:
    def __init__(
        self,
        stack: GeneralizedActionFixedStack,
        max_actions: torch.Tensor,  # (batch_size,)
        padding_idx: int,  # idx for <PAD> action.
        start_empty: bool,
    ):
        super(BeamItems, self).__init__()

        self.batch_size = stack.batch_size
        self.beam_size = stack.beam_size
        self.sample_size = stack.sample_size
        self.stack = stack

        self.scores = (
            torch.tensor([-float("inf")], device=stack.hiddens.device)
            .expand(self.batch_size, self.sample_size, self.beam_size)
            .clone()
        )
        # Log probs of lastly shifted token.
        self.last_token_log_probs = (
            torch.tensor([-float("inf")], device=stack.hiddens.device)
            .expand(self.batch_size, self.sample_size, self.beam_size)
            .clone()
        )
        # Fill the first element with 0.
        if not start_empty:
            self.scores[..., 0] = 0
            self.last_token_log_probs[..., 0] = 0

        self.action_path = ActionPath(
            batch_size=self.batch_size,
            beam_size=self.beam_size,
            sample_size=self.sample_size,
            max_actions=max_actions,
            padding_idx=padding_idx,
            device=stack.hiddens.device,
        )

        # Beams should not be empty at first (i.e., must have at least size 1).
        if not start_empty:
            self.active_widths = self.scores.new_ones((self.batch_size, self.sample_size), dtype=torch.long)
        else:
            self.active_widths = self.scores.new_zeros((self.batch_size, self.sample_size), dtype=torch.long)

    @property
    def actions(self):
        return self.action_path.actions

    @property
    def actions_pos(self):
        return self.action_path.actions_pos

    def active_idxs(self) -> tuple[torch.Tensor, ...]:
        """
        :return (batch_idxs, sample_idxs, beam_idxs): All active idxs according to active beam sizes for each batch and sample defined by self.beam_widths.
        """
        return self.active_idx_mask().nonzero(as_tuple=True)

    def active_idx_mask(self) -> torch.Tensor:
        order = torch.arange(self.beam_size, device=self.active_widths.device)
        return order < self.active_widths.unsqueeze(-1)

    def move_elements(
        self,
        source: "BeamItems",
        self_idxs: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],  # (batch_indices, sample_indicies, beam_indices)
        source_idxs: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
        # new_scores=None,
    ):
        self.scores[self_idxs] = source.scores[source_idxs]
        self.last_token_log_probs[self_idxs] = source.last_token_log_probs[source_idxs]

        self.stack.move_beams(self_idxs, source.stack, source_idxs)
        self.action_path.move_beams(self_idxs, source.action_path, source_idxs)

    def do_action(
        self,
        actions: torch.Tensor,  # (batch_size, sample_size, beam_size)
    ):
        # Update action_path.
        self.action_path.add(actions, self.active_idxs())

    def get_best_item_mask(
        self,
    ) -> torch.Tensor:  # (batch_size, sample_size, beam_size)
        inactive_mask = torch.arange(self.beam_size, dtype=torch.long, device=self.scores.device).unsqueeze(0).expand(
            self.batch_size, self.sample_size, -1
        ) >= self.active_widths.unsqueeze(-1)

        scores = self.scores.clone().detach()

        # Set -inf for inactive and inactive beam items.
        scores[inactive_mask] = -float("inf")

        best_beam_item_idxs = scores.argmax(dim=-1, keepdim=True)
        assert best_beam_item_idxs.size() == (self.batch_size, self.sample_size, 1)

        # Calculate best action mask.
        best_beam_item_mask = (
            torch.arange(self.beam_size, device=self.scores.device).view(1, 1, self.beam_size) == best_beam_item_idxs
        )
        assert best_beam_item_mask.size() == (
            self.batch_size,
            self.sample_size,
            self.beam_size,
        )

        # debug
        # logger.warning(f"{self.scores=}")
        # logger.warning(f"{best_beam_item_mask=}")
        # debug

        return best_beam_item_mask

    # Mainly for debug.
    def get_beam_actions(self, action_dict: GeneralizedActionDict) -> list[list[list[list[str]]]]:
        beam_actions = [
            [
                [
                    [
                        action_dict.i2a(a_id)
                        for a_id in self.actions[
                            batch_i,
                            sample_i,
                            beam_i,
                            : self.actions_pos[batch_i, sample_i, beam_i] + 1,
                        ]
                    ]
                    for beam_i in range(self.beam_size)
                ]
                for sample_i in range(self.sample_size)
            ]
            for batch_i in range(self.batch_size)
        ]

        return beam_actions


class GeneralizedActionRNNGCell(nn.Module):
    """
    RNNGCell receives next action and input word embedding, do action, and returns next updated hidden states.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        vocab_size: int,
        vocab_padding_idx: int,
        dropout: float,
        layernorm: bool,
        action_dict: GeneralizedActionDict,
        attention_composition: bool,
    ):
        super(GeneralizedActionRNNGCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Set regularizer.
        if layernorm:
            self.initial_emb_regularizer = nn.LayerNorm(normalized_shape=input_size)
            self.nt_emb_regularizer = nn.LayerNorm(normalized_shape=input_size)
            self.token_emb_regularizer = nn.LayerNorm(normalized_shape=input_size)
            self.output_regularizer = nn.LayerNorm(normalized_shape=hidden_size)
        else:
            dropout_layer = nn.Dropout(dropout, inplace=True)
            self.initial_emb_regularizer = dropout_layer
            self.nt_emb_regularizer = dropout_layer
            self.token_emb_regularizer = dropout_layer
            self.output_regularizer = dropout_layer

        self.nt_emb = nn.Sequential(nn.Embedding(action_dict.num_nts, input_size), self.nt_emb_regularizer)
        self.token_emb = nn.Sequential(
            nn.Embedding(
                vocab_size,
                input_size,
                padding_idx=vocab_padding_idx,
            ),
            self.token_emb_regularizer,
        )
        self.stack_rnn = MultiLayerLSTMCell(input_size, hidden_size, num_layers, dropout=dropout, layernorm=layernorm)
        self.output = nn.Sequential(self.output_regularizer, nn.Linear(hidden_size, input_size), nn.ReLU())
        self.composition = (
            AttentionComposition(
                w_dim=input_size,
                dropout=dropout,
                layernorm=layernorm,
                num_labels=action_dict.num_nts,
            )
            if attention_composition
            else LSTMComposition(dim=input_size, dropout=dropout, layernorm=layernorm)
        )

        self.initial_emb = nn.Sequential(nn.Embedding(1, input_size), self.initial_emb_regularizer)

        self.action_dict = action_dict

    def get_initial_hidden(self, x) -> torch.Tensor:
        """
        x: (batch_size, sent_len?)
        Return: hidden and cell: [(batch_size, hidden_size, num_layers), (batch_size, hidden_size, num_layers)]

        Note that the returned hidden_states do not need sample/beam dimension because the same values are used.
        """
        iemb = self.initial_emb(x.new_zeros(x.size(0), dtype=torch.long))  # (batch_size, input_size)
        return self.stack_rnn(iemb, None)

    def forward(
        self,
        x: torch.Tensor,  # (batch_size, sent_len)
        actions: torch.Tensor,  # (batch_size, sample_size, beam_size)
        stack: GeneralizedActionFixedStack,
    ):
        """
        Similar to update_stack_rnn.

        :param word_vecs: (batch_size, sent_len, input_size)
        :param actions: (batch_size, 1)
        """

        reduce_batches = (actions == self.action_dict.reduce_idx).nonzero(as_tuple=True)
        nt_batches = ((self.action_dict.nt_begin_idx <= actions) * (actions <= self.action_dict.nt_end_idx)).nonzero(
            as_tuple=True
        )
        shift_batches = (actions == self.action_dict.shift_idx).nonzero(as_tuple=True)

        no_nop_batches = (actions != self.action_dict.padding_idx).nonzero(as_tuple=True)

        # debug
        # logger.warning(f"{actions.size()=}")
        # logger.warning(f"{no_nop_batches=}")
        # debug

        new_input = stack.trees.new_zeros(
            stack.hiddens.size()[:-3] + (self.input_size,)
        )  # (batch_size, sample_size, beam_size, input_size)

        # First fill in trees. Then, gather those added elements in a column, which become
        # the input to stack_rnn.
        if shift_batches[0].size(0) > 0:
            token_ids = x[shift_batches[:1] + (stack.pointer[shift_batches],)]
            token_embs = self.token_emb(token_ids).to(new_input.dtype)
            stack.do_shift(shift_batches, token_embs)
            new_input[shift_batches] = token_embs

        if nt_batches[0].size(0) > 0:
            nt_ids = (actions[nt_batches] - self.action_dict.nt_begin_idx) // (self.action_dict.num_actions_for_each_nt)
            nt_pos = (actions[nt_batches] - self.action_dict.nt_begin_idx) % (self.action_dict.num_actions_for_each_nt)
            nt_embs = self.nt_emb(nt_ids).to(new_input.dtype)
            stack.do_nt(nt_batches, nt_embs, nt_ids, nt_pos)
            new_input[nt_batches] = nt_embs

        if reduce_batches[0].size(0) > 0:
            children, ch_lengths, reduced_nt, reduced_nt_ids = stack.collect_reduced_children(reduce_batches)

            if isinstance(self.composition, AttentionComposition):
                hidden_head = stack.hidden_head(batches=reduce_batches)[
                    :, :, -1
                ]  # The return of stack.hidden_head has the size (batches, hidden_size, num_layers)
                stack_h = self.output(hidden_head)
            else:
                stack_h = None
            new_child, _, _ = self.composition(children, ch_lengths, reduced_nt, reduced_nt_ids, stack_h)
            stack.do_reduce(reduce_batches, new_child)
            new_input[reduce_batches] = new_child.to(new_input.dtype)

        # Input for rnn should be (beam_size, input_size). During beam search, new_input has different size.
        new_hidden, new_cell = self.stack_rnn(
            new_input[no_nop_batches],
            (stack.hidden_head(offset=1, batches=no_nop_batches), stack.cell_head(offset=1, batches=no_nop_batches)),
        )

        # Do nothing for nop actions (e.g., PAD).
        stack.update_hidden(new_hidden, new_cell, no_nop_batches)

        # The shape of stack.hidden_head() is (batch_size, hidden_size, num_layers)
        return stack.hidden_head()[..., -1]  # (batch_size, hidden_size)


# Deterministic version of sample_beam_items.
def sample_beam_items_deterministic(
    flattened_logits: torch.Tensor,  # (B*, M) where M is the number of all candidates.
    max_num_to_sample: int,
) -> torch.Tensor:  # Ordered idices of sampled items of size (B*, N) where N <= num_to_sample (N < num_to_sample if there are few candidates).
    # Get sizes.
    # Sample top-k.
    # Simply deterministically take the top-k largetst items.
    sampled_items = flattened_logits.topk(k=max_num_to_sample, dim=-1).indices

    return sampled_items


@torch.compile
def get_until_kth_mask(
    input: torch.Tensor,  # (*, N)
    k_tensor: torch.Tensor,  # (*,)
) -> torch.Tensor:  # (*, N)
    """"""
    int_input = input.int() * 2
    count_so_far = int_input.cumsum_(dim=-1)
    added_count_so_far = torch.where(
        condition=input, input=count_so_far, other=count_so_far + 1
    )  # This is necessary to avoid giving True to the elements on the right of k-th True element in input.
    return added_count_so_far <= (k_tensor.unsqueeze(-1) * 2)


def get_finished_word_sync_step_batches(
    open_beam: BeamItems,
    step_complete_beam: BeamItems,
    word_sync_step_limit: int,
    tmp_step_count: int,
) -> torch.Tensor:
    finished_word_sync_step_batches = (
        (open_beam.active_widths == 0)  # In case there is no candidates.
        + (
            step_complete_beam.active_widths >= step_complete_beam.beam_size
        )  # In case the step_complete_beam is already full.
        + (tmp_step_count + 1 >= word_sync_step_limit)
        * (
            step_complete_beam.active_widths > 0
        )  # In case inner steps exceeds the limit and step_complete_beam is not empty except when all tokens are already shifted (i.e., the step complete action is FINISH).
    )
    return finished_word_sync_step_batches


class GeneralizedActionFixedStackRNNG(nn.Module):
    def __init__(
        self,
        action_dict: GeneralizedActionDict,
        vocab_size: int = 100,
        vocab_padding_idx: int = 0,
        w_dim: int = 20,
        h_dim: int = 20,
        num_layers: int = 1,
        dropout: float = 0.0,
        layernorm: bool = True,
        attention_composition: bool = False,
    ):
        super(GeneralizedActionFixedStackRNNG, self).__init__()
        self.action_dict = action_dict
        self.vocab_padding_idx = vocab_padding_idx

        # Set regularizer.
        if layernorm:
            self.emb_regularizer = nn.LayerNorm(normalized_shape=w_dim)
        else:
            dropout_layer = nn.Dropout(dropout, inplace=True)
            self.emb_regularizer = dropout_layer

        self.vocab_size = vocab_size

        self.rnng = GeneralizedActionRNNGCell(
            input_size=w_dim,
            hidden_size=h_dim,
            num_layers=num_layers,
            vocab_size=vocab_size,
            vocab_padding_idx=vocab_padding_idx,
            dropout=dropout,
            layernorm=layernorm,
            action_dict=self.action_dict,
            attention_composition=attention_composition,
        )

        self.vocab_mlp = nn.Linear(w_dim, vocab_size)
        self.num_layers = num_layers
        self.action_size = action_dict.action_size
        self.action_mlp = nn.Linear(w_dim, self.action_size)
        self.input_size = w_dim
        self.hidden_size = h_dim

        self.vocab_mlp.weight = self.rnng.token_emb[0].weight

    def get_next_action_candidates(
        self,
        beam: BeamItems,
        sent_lengths: torch.Tensor,  # (batch_size,)
        token_ids: torch.Tensor,  # (batch_size, sent_len)
        finished_word_sync_step_batches: torch.Tensor,  # (batch_size, sample_size) this is necessary to avoid applying word_sync_step_limit to FINISH step.
        word_sync_step: int,
    ) -> tuple[
        torch.Tensor,  # (batch_size, sample_size, beam_size, action_size) each entry has the score (log probs) for next actions.
        torch.Tensor,  # (batch_size, sample_size, beam_size, action_size) token log probs.
    ]:
        # Calculate scores for next action candidates.
        # Take the hidden vector of the last layer.
        # Note that the size of hidden_head() is (batches, hidden_size, num_layers).
        # Here, batches is the same as batch_size * sample_size * beam_size.
        # TODO: deal with beam width for efficiency...?
        # But in practice, almost all of the beams are filled by just after one inference step (because there are many possible next actions).
        hiddens = self.rnng.output(beam.stack.hidden_head()[:, :, -1])

        # Calculate invalid action masks.
        invalid_action_mask = self.get_invalid_action_mask(
            beam=beam,
            sent_lengths=sent_lengths,
        )  # (batch_size, sample_size, beam_size, num_actions)

        # Calcualte action logits.
        action_logits: torch.Tensor = (
            self.action_mlp(hiddens).view(beam.batch_size, beam.sample_size, beam.beam_size, -1)
            # .float()
        )

        # This is necessary since the log_softmax of inactive beams would be NAN (because all actions are invalid and comes with -inf logits)
        # Somehow, we need to use torch.where because the gradint calculation does not work when the output of log_soft_max is overwritten(?) (not really sure though).

        # Here, to align with supervised training loss, we apply invalid_action_mask after calculating log probs.
        next_action_log_probs = torch.where(
            condition=(
                finished_word_sync_step_batches.logical_not().view(beam.batch_size, beam.sample_size, 1)
                * beam.active_idx_mask()
            )
            .unsqueeze(-1)
            .expand(-1, -1, -1, self.action_dict.action_size)
            * invalid_action_mask.logical_not(),
            input=torch.nn.functional.log_softmax(input=action_logits, dim=-1),
            other=-float("inf"),
        )

        # debug
        assert (
            finished_word_sync_step_batches.logical_not().view(beam.batch_size, beam.sample_size, 1)
            * beam.active_idx_mask()
        ).unsqueeze(-1).expand(-1, -1, -1, self.action_dict.action_size).size() == invalid_action_mask.size()
        # debug

        # Next, calcualte token prediction log probs.
        # Calculate token probabilities.
        assert word_sync_step < token_ids.size(1)
        next_tokens = token_ids[:, word_sync_step]

        # Simply inserting -inf to padding_idx does not work because the substitution blocks the gradient flow.
        token_logits = self.vocab_mlp(hiddens).view(beam.batch_size, beam.sample_size, beam.beam_size, -1)
        token_logits[..., self.vocab_padding_idx] = -float("inf")

        token_neg_ll = torch.nn.functional.cross_entropy(
            input=token_logits.view(beam.batch_size * beam.sample_size * beam.beam_size, self.vocab_size),
            target=next_tokens.view(beam.batch_size, 1, 1).repeat(1, beam.sample_size, beam.beam_size).view(-1),
            reduction="none",
            ignore_index=self.vocab_padding_idx,
        ).view(beam.batch_size, beam.sample_size, beam.beam_size)

        token_log_probs = -token_neg_ll

        # In-place operation.
        next_action_log_probs.index_add_(
            dim=-1,
            index=torch.tensor(
                [self.action_dict.shift_idx], dtype=torch.long, device=token_ids.device
            ),  # BTW, this index must not be a scalar tensor (but a single dimension tensor) when one wants to compile a function containing this part.
            source=token_log_probs.unsqueeze(-1),
        )

        # return next_action_log_probs, token_log_probs, torch.tensor([])
        return next_action_log_probs, token_log_probs

    def step_word_sync_beam_search(
        self,
        x: torch.Tensor,  # (batch_size, sent_len)
        open_beam: BeamItems,
        step_complete_beam: BeamItems,
        sent_lengths: torch.Tensor,  # (batch_size,)
        min_shift_size: int,
        finished_word_sync_step_batches: torch.Tensor,  # (batch_size, sample_size) this is necessary to avoid applying word_sync_step_limit to FINISH step.
        word_sync_step: int,
    ):
        # TODO: only consider unfinished items (i.e., step_completed_beam.active_widths < beam_widths) for efficiency.

        # Only use step_not_completed_beam here (because next action candidates are only taken from the items in step_not_completed_beam).
        next_action_log_probs, token_log_probs = self.get_next_action_candidates(
            beam=open_beam,
            sent_lengths=sent_lengths,
            token_ids=x,
            finished_word_sync_step_batches=finished_word_sync_step_batches,
            word_sync_step=word_sync_step,
        )

        assert next_action_log_probs.size() == (
            open_beam.batch_size,
            open_beam.sample_size,
            open_beam.beam_size,
            self.action_dict.action_size,
        )

        # Calcualte the scores used for beam transition (current beam item score + next action score).
        next_beam_item_candidate_logits = open_beam.scores.unsqueeze(dim=-1) + next_action_log_probs
        assert next_beam_item_candidate_logits.size() == (
            open_beam.batch_size,
            open_beam.sample_size,
            open_beam.beam_size,
            self.action_dict.action_size,
        )

        # Choose next beam items.

        # Next, enumerate and sample only SHIFT actions.
        # No need to explicitly flatten the candidates, because each beam item can have at most one shift action.
        flattened_complete_action_candidate_logits = next_beam_item_candidate_logits[..., self.action_dict.shift_idx]
        assert flattened_complete_action_candidate_logits.size() == (
            open_beam.batch_size,
            open_beam.sample_size,
            open_beam.beam_size,
        )

        # Calculate number of shift actions forced to sample.
        num_force_sample = torch.minimum(
            torch.minimum(
                step_complete_beam.beam_size - step_complete_beam.active_widths,
                torch.tensor(min_shift_size, dtype=torch.long, device=x.device),
            ),
            # Need to use the logits before clamp.
            (next_beam_item_candidate_logits[..., self.action_dict.shift_idx] != -float("inf")).count_nonzero(dim=-1),
        )
        max_num_force_sample = num_force_sample.max().item()

        # Force to sample actions to complete word sync beam step.
        tmp_force_sampled_items = sample_beam_items_deterministic(
            flattened_logits=flattened_complete_action_candidate_logits, max_num_to_sample=max_num_force_sample
        )

        # Recover the idxs for the full flattened candidates.
        force_sampled_items = self.action_dict.shift_idx + tmp_force_sampled_items * self.action_dict.action_size

        # Next, sample from remaining candidates.

        # First, disable the logits for already sampled complete actions.
        # TODO: this can be simpler by counting scores not equal to -float('inf')...?
        force_num_sampled_mask = torch.arange(max_num_force_sample, device=next_action_log_probs.device).view(
            1, 1, max_num_force_sample
        ) < num_force_sample.unsqueeze(-1)

        force_sampled_batches = force_num_sampled_mask.nonzero(as_tuple=True)[:-1] + (
            tmp_force_sampled_items[force_num_sampled_mask],
        )

        # TODO: this clone may not be necessary; this can be made simpler by using torch.scatter?
        tmp_other_candidate_logits = next_beam_item_candidate_logits.clone()
        tmp_other_candidate_logits[..., self.action_dict.shift_idx][force_sampled_batches] = -float("inf")
        flattened_other_candidate_logits = tmp_other_candidate_logits.view(
            open_beam.batch_size, open_beam.sample_size, open_beam.beam_size * self.action_dict.action_size
        )

        other_active_candidate_mask = torch.arange(open_beam.beam_size, dtype=torch.long, device=x.device).view(
            1,
            1,
            open_beam.beam_size,
        ) < (flattened_other_candidate_logits != -float("inf")).count_nonzero(dim=-1).unsqueeze(-1)

        # Sample from remaining candidates.
        tmp_other_sampled_items = sample_beam_items_deterministic(
            flattened_logits=flattened_other_candidate_logits, max_num_to_sample=open_beam.beam_size
        )
        num_other_samplable_complete_actions = step_complete_beam.beam_size - (
            step_complete_beam.active_widths + num_force_sample
        )
        assert (num_other_samplable_complete_actions >= 0).all()

        tmp_other_action_ids = tmp_other_sampled_items % self.action_dict.action_size

        other_samplable_complete_action_mask = get_until_kth_mask(
            input=(
                tmp_other_action_ids
                == self.action_dict.shift_idx  # This does not consider whether the candidate is active, but this should be fine.
            ),  # Mask for sampled complete actions.
            k_tensor=num_other_samplable_complete_actions,  # Number of complete actions that can be sampled.
        )

        other_num_sampled_mask = other_active_candidate_mask * other_samplable_complete_action_mask

        # Next, arange sampled items by combining shift actions forced to sample and actions sampled from remaining candidates.
        tmp_sampled_items = torch.cat((force_sampled_items, tmp_other_sampled_items), dim=-1)
        assert tmp_sampled_items.size() == (
            open_beam.batch_size,
            open_beam.sample_size,
            max_num_force_sample + open_beam.beam_size,
        )
        tmp_action_ids = tmp_sampled_items % self.action_dict.action_size

        sampled_mask = torch.cat(
            (force_num_sampled_mask, other_num_sampled_mask), dim=-1
        )  # Masking really sampled items (sampled_items itself may contain items not sampeld).
        assert sampled_mask.size() == (
            open_beam.batch_size,
            open_beam.sample_size,
            max_num_force_sample + open_beam.beam_size,
        )

        complete_action_mask = tmp_action_ids == self.action_dict.shift_idx
        assert complete_action_mask.size() == (
            open_beam.batch_size,
            open_beam.sample_size,
            max_num_force_sample + open_beam.beam_size,
        )

        # Move beam items.
        # Note that beam item move must be first be done for step_complete_beam before open_beam (otherwise the source items for step_completed might be overwritten before copied).
        tmp_source_beam_ids = tmp_sampled_items // self.action_dict.action_size

        # First, calculate destination/source idxs for moving beam items for step_completed_beam.
        step_complete_mask = sampled_mask * complete_action_mask
        step_complete_source_beam_idxs = step_complete_mask.nonzero(as_tuple=True)[:-1] + (
            tmp_source_beam_ids[step_complete_mask],
        )

        step_complete_new_active_widths = step_complete_beam.active_widths + step_complete_mask.count_nonzero(dim=-1)
        step_complete_beam_order = torch.arange(step_complete_beam.beam_size, dtype=torch.long, device=x.device).view(
            1, 1, step_complete_beam.beam_size
        )
        step_complete_target_idxs = (
            (step_complete_beam.active_widths.unsqueeze(-1) <= step_complete_beam_order)
            * (step_complete_beam_order < step_complete_new_active_widths.unsqueeze(-1))
        ).nonzero(as_tuple=True)

        # Update beam widths and move beam items.
        step_complete_beam.active_widths = step_complete_new_active_widths
        step_complete_beam.move_elements(
            source=open_beam,  # Note that the source is open_beam not step_complete_beam.
            self_idxs=step_complete_target_idxs,
            source_idxs=step_complete_source_beam_idxs,
        )

        # Update scores, total_action/token_neg_ll.
        step_complete_source_item_idxs = step_complete_source_beam_idxs[:-1] + (tmp_sampled_items[step_complete_mask],)

        # Update scores.
        step_complete_beam.scores[step_complete_target_idxs] += next_action_log_probs.view(
            step_complete_beam.batch_size,
            step_complete_beam.sample_size,
            step_complete_beam.beam_size * self.action_dict.action_size,
        )[step_complete_source_item_idxs]

        # Do not add but simply update the token probability by the prediction just made.
        step_complete_beam.last_token_log_probs[step_complete_target_idxs] = token_log_probs[
            step_complete_source_beam_idxs
        ]

        ####
        # Model and stack is updated later for step_complte_beam when all items complete a word sync beam step.
        ####

        # Next, calculate destination/source idxs for moving beam items for open_beam.
        open_mask = sampled_mask * complete_action_mask.logical_not()
        open_source_beam_idxs = (open_mask.nonzero(as_tuple=True))[:-1] + (tmp_source_beam_ids[open_mask],)

        open_new_active_widths = open_mask.count_nonzero(dim=-1)
        assert (open_new_active_widths <= open_beam.beam_size).all()
        open_target_idxs = (
            torch.arange(open_beam.beam_size, dtype=torch.long, device=x.device).view(1, 1, open_beam.beam_size)
            < open_new_active_widths.unsqueeze(-1)
        ).nonzero(as_tuple=True)

        # Update beam widths and move beam items.
        open_beam.active_widths = open_new_active_widths
        open_beam.move_elements(source=open_beam, self_idxs=open_target_idxs, source_idxs=open_source_beam_idxs)

        # Update scores.
        open_source_item_idxs = open_source_beam_idxs[:-1] + (tmp_sampled_items[open_mask],)

        # Update scores.
        open_beam.scores[open_target_idxs] += next_action_log_probs.view(
            open_beam.batch_size, open_beam.sample_size, open_beam.beam_size * self.action_dict.action_size
        )[open_source_item_idxs]

        # Update stack.
        open_next_action_ids = torch.full(
            size=(open_beam.batch_size, open_beam.sample_size, open_beam.beam_size),
            fill_value=self.action_dict.padding_idx,
            device=x.device,
        )
        open_next_action_ids[open_target_idxs] = tmp_action_ids[open_mask]

        # debug
        # print(f"{open_next_action_ids=}")
        # print(f"{open_source_beam_idxs=}")
        # debug

        self.rnng(x=x, actions=open_next_action_ids, stack=open_beam.stack)

        # Update beam states.
        open_beam.do_action(
            actions=open_next_action_ids,
        )

    def complete_word_sync_step(
        self,
        x: torch.Tensor,  # (batch_size, sent_len)
        open_beam: BeamItems,
        step_complete_beam: BeamItems,
        sent_lengths: torch.Tensor,  # (batch_size,)
        word_sync_step: int,
    ) -> tuple[
        torch.Tensor,  # (batch_size) Active beam width before move.
        torch.Tensor,  # (batch_size, sample_size, beam_size) Scores of action sequences in the beam before token log probability is added.
        torch.Tensor,  # (batch_size, sample_size, beam_size) Scores of token
    ]:
        # Get the scores of action sequences ending with SHIFT.
        # Note that, the scores of beam items contain token log probs, so we need to subract them from the scores.
        beam_scores_before_token_prediction = step_complete_beam.scores - step_complete_beam.last_token_log_probs
        active_widths_before_move = step_complete_beam.active_widths.clone().detach()
        token_log_probs = step_complete_beam.last_token_log_probs.clone().detach()

        # First, get action ids for completing the word sync step.
        complete_action_ids = torch.full(
            size=(step_complete_beam.batch_size, step_complete_beam.sample_size, step_complete_beam.beam_size),
            fill_value=self.action_dict.padding_idx,
            device=x.device,
        )

        active_mask = step_complete_beam.active_idx_mask()
        to_shift_batch_mask = word_sync_step < sent_lengths
        active_to_shift_mask = to_shift_batch_mask.view(step_complete_beam.batch_size, 1, 1) * active_mask

        complete_action_ids[active_to_shift_mask] = self.action_dict.shift_idx

        # Next, update stack.
        self.rnng(x=x, actions=complete_action_ids, stack=step_complete_beam.stack)

        # Update beam states.
        step_complete_beam.do_action(
            actions=complete_action_ids,
        )

        # Move unifinished items from step_complete_beam to open_beam.

        assert (step_complete_beam.active_widths <= open_beam.beam_size).all()

        shift_batch_mask = (word_sync_step + 1) < sent_lengths  # Non-finishing shifts.
        finish_batch_mask = sent_lengths == (word_sync_step + 1)

        move_to_next_step_mask = active_mask * shift_batch_mask.view(-1, 1, 1)
        source_idxs = move_to_next_step_mask.nonzero(as_tuple=True)

        # Since we assume the active width of step_complete_beam is smaller than the beam size of open_beam, the target idxs and source idxs are the same. In other words, we can simply copy the elements to the same positions.
        target_idxs = source_idxs

        open_beam.move_elements(source=step_complete_beam, self_idxs=target_idxs, source_idxs=source_idxs)

        # Reset beam active_widths.

        # For finished batches, simply deactivate open beams.
        open_beam.active_widths[finish_batch_mask] = 0

        # For unfinished batches, update active_widths.
        open_beam.active_widths[shift_batch_mask] = step_complete_beam.active_widths[shift_batch_mask]
        step_complete_beam.active_widths[shift_batch_mask] = 0

        # Do nothing for already-finished items (with PAD action).

        # Return beam scores (before token prediction) and token log probability.
        return active_widths_before_move, beam_scores_before_token_prediction, token_log_probs

    def count_nt_pos(
        self,
        actions: torch.Tensor,  # (batch_size, N*, episode_len)
    ) -> list[int]:  # List containing nt_pos counts.
        """Simply count nt insert positions.

        Note that this is very rough way to analyze strategy. For example, nt insert positions of bottom-up strategy differs depending on phrase size, so you cannot know whether an NT action is bottom-up or not only by looking at nt insert positions.
        But, this method is efficient during training.
        """

        nt_actions = (self.action_dict.nt_begin_idx <= actions).nonzero(as_tuple=True)
        nt_pos = (actions[nt_actions] - self.action_dict.nt_begin_idx) % self.action_dict.num_actions_for_each_nt

        return nt_pos.bincount().tolist()

    def build_new_beam(
        self,
        initial_hidden: torch.Tensor,  # (batch_size, input_size)
        batch_size: int,
        beam_size: int,
        sample_size: int,
        stack_size: int,
        max_actions: torch.Tensor,  # (batch_size,)
        start_empty: bool,
    ) -> BeamItems:
        stack = GeneralizedActionFixedStack(
            initial_hidden=initial_hidden,
            stack_size=stack_size,
            input_size=self.input_size,
            batch_size=batch_size,
            beam_size=beam_size,
            sample_size=sample_size,
        )

        # Initialize beam.
        beam = BeamItems(
            stack=stack,
            max_actions=max_actions,
            padding_idx=self.action_dict.padding_idx,
            start_empty=start_empty,
        )
        return beam

    def get_invalid_action_mask(
        self,
        beam: BeamItems,
        sent_lengths: torch.Tensor,  # (batch_size,)
    ) -> torch.Tensor:  # (batch_size, sample_size, beam_size, num_actions)
        """This is to ensure that the actions generate correct number of tokens (if stack size is large enough (i.e., larger than number of tokens)); but this does not ensure that the actions form a tree."""

        action_order = torch.arange(self.action_size, device=sent_lengths.device)

        sent_lengths = sent_lengths.unsqueeze(-1)  # add beam dimension

        batch_size = sent_lengths.size(0)

        # Note the offset -1 of nt_index_pos.
        nopen_parens = beam.stack.nt_index_pos + 1

        # Naive: is_stack_top_open_nt may be obtained by just view...? (is_stack_top_open_nt_submask is an advanced index)
        # Note the offset of stack.nt_index.
        is_stack_top_open_nt_submask = (
            # Note the offset of nt_index and top_position.
            beam.stack.nt_index[beam.stack.batch_index + (beam.stack.nt_index_pos[beam.stack.batch_index],)] - 1
            == beam.stack.top_position[beam.stack.batch_index] - 1
        ) * (beam.stack.nt_index_pos[beam.stack.batch_index] > -1)
        is_stack_top_open_nt = torch.tensor([False], dtype=torch.bool, device=sent_lengths.device).repeat(
            batch_size, beam.sample_size, beam.beam_size
        )
        is_stack_top_open_nt[
            (
                beam.stack.batch_index[0][is_stack_top_open_nt_submask],
                beam.stack.batch_index[1][is_stack_top_open_nt_submask],
                beam.stack.batch_index[2][is_stack_top_open_nt_submask],
            )
        ] = True
        stack_top_nt_index = torch.full(
            (batch_size, beam.sample_size, beam.beam_size),
            fill_value=-1,
            dtype=torch.long,
            device=sent_lengths.device,
        )
        exists_open_nt_batches = (beam.stack.nt_index_pos > -1).nonzero(as_tuple=True)
        stack_top_nt_index[exists_open_nt_batches] = (
            beam.stack.nt_index[exists_open_nt_batches + (beam.stack.nt_index_pos[exists_open_nt_batches],)] - 1
        )

        # reduce_mask[i,j,k]=True means k is a not allowed reduce action for (i,j).
        reduce_mask = (action_order == self.action_dict.reduce_idx).view(1, 1, 1, -1)
        reduce_mask = ((nopen_parens == 0) + (is_stack_top_open_nt)).unsqueeze(-1) * reduce_mask

        nt_pos = ((action_order - self.action_dict.nt_begin_idx) % self.action_dict.num_actions_for_each_nt).view(
            1, 1, 1, -1
        )

        # Check the storage of beam.actions, which is bounded beforehand.
        # Theoretically +1 seems sufficient (for rhs); extra +2 is for saving cases
        # where other actions (reduce/shift) are prohibited for some reasons.
        # Remaining actions = max_actions - words_to_shift - open_nts_to_reduce - finish
        remaining_actions = (
            (
                beam.action_path.max_actions.view(-1, 1, 1) - beam.actions_pos
            )  # Consider different max_actions for different batches.
            - (sent_lengths.view(-1, 1, 1) - beam.stack.pointer)
            - nopen_parens
            - 1
        )
        # Also need to consider REDUCE.
        allowed_nts = remaining_actions // 2

        # nt_mask[i,j,k]=True means k is a not allowed nt action for (i,j).
        nt_mask = (
            (self.action_dict.nt_begin_idx <= action_order) * (action_order <= self.action_dict.nt_end_idx)
        ).view(1, 1, 1, -1)

        nt_mask = (
            # Consider (in)valid insert positions.
            # NTs can only be inserted to the left of complete nodes that are above top-most open nt.
            (
                # Note that stack_top_nt_index is initialized with -1, so the following is valid even when there is no open nt.
                nt_pos > (beam.stack.top_position - (stack_top_nt_index + 1)).unsqueeze(-1)
            )
            # Consider the resource limits.
            + (allowed_nts <= 0).unsqueeze(-1)
            +
            # Heuristics
            # Check the storage of fixed stack size.
            (
                (
                    # For nts in stack top, we need minimally two additional elements to process arbitrary future structure.
                    (beam.stack.top_position > beam.stack.stack_size - 2)
                    # Besides, top-down NT (i.e., NT(X;0)) is not possible when all words (except eos token) are already shifted.
                    + (beam.stack.pointer == (sent_lengths - 1).view(-1, 1, 1))
                ).unsqueeze(-1)
                * (
                    (nt_pos == 0)
                    +
                    # Also, non-top-down NT (when there is open nt on stack top) requires two stack elements to be completed.
                    (nt_pos > 0) * is_stack_top_open_nt.unsqueeze(-1)
                )
            )
            +
            # For nts inserted not to stack top, we need minimally one additional elements to process arbittrary future structure.
            ((beam.stack.top_position > beam.stack.stack_size - 1).unsqueeze(-1) * (nt_pos > 0))
        ) * nt_mask

        shift_mask = (action_order == self.action_dict.shift_idx).view(1, 1, 1, -1)
        shift_mask = (
            # Heuristics to deal with fixed stack size.
            (
                # when nopen=0, shift accompanies nt, thus requires two.
                ((nopen_parens == 0) * (beam.stack.top_position > beam.stack.stack_size - 2))
                +
                # otherwise, requires one room.
                ((nopen_parens > 0) * (beam.stack.top_position > beam.stack.stack_size - 1))
            )
            +
            # Shifting eos token.
            # EOS token cannot be shift until all open nts are reduced.
            (beam.stack.pointer == (sent_lengths - 1).view(-1, 1, 1)) * (nopen_parens > 0)
        ).unsqueeze(-1) * shift_mask

        # <PAD> is invalid unless eos token (i.e., the last token) is shifted.
        pad_mask = (action_order == self.action_dict.padding_idx).view(1, 1, 1, -1)
        pad_mask = (beam.stack.pointer < sent_lengths.view(-1, 1, 1)).unsqueeze(-1) * pad_mask

        # Disable all actions except pad when eos token has been shifted.
        except_pad_mask = (action_order != self.action_dict.padding_idx).view(1, 1, 1, -1) * (
            beam.stack.pointer == sent_lengths.view(-1, 1, 1)
        ).unsqueeze(-1)

        beam_inactive_mask = (
            torch.arange(beam.beam_size, device=reduce_mask.device).view(1, 1, -1) >= beam.active_widths.unsqueeze(-1)
        ).unsqueeze(-1)

        invalid_action_mask = (
            reduce_mask + nt_mask + shift_mask + pad_mask + except_pad_mask + beam_inactive_mask
        )  # (batch_size, sample_size, beam_size, num_actions)

        # Check if there is at least one valid action for active beams.
        # Add True for inactive beams for this purpose (a bit tricky).
        if not ((invalid_action_mask.logical_not() + beam_inactive_mask).count_nonzero(dim=-1) > 0).all():
            logger.warning("No valid next action exists for some batch/sample/beam!!!!!!")
            raise NoValidNextActionError

        return invalid_action_mask

    def inference(
        self,
        x: torch.Tensor,  # (batch_size, sent_len)
        stack_size: int,
        beam_size: int,  # For now, we force to use the same beam size for both open/step_complete_beam.
        min_shift_size: int,
        word_sync_step_limit: int,
    ) -> tuple[
        list[list[int]],  # (batch_size, action_len) Best action sequences in the final beam; paddings are removed.
        list[list[list[float]]],  # (batch_size, sent_len, beam_size) Beam item scores at each word sync step.
        list[
            list[list[float]]
        ],  # (batch_size, sent_len, beam_size) Next token log probability of each beam item at each word sync step.
    ]:
        # Get sizes.
        batch_size: int = x.size(0)
        sent_lengths = (x != self.vocab_padding_idx).sum(dim=-1)
        device = x.device

        max_actions: torch.Tensor = torch.maximum(sent_lengths * 3, sent_lengths + 3)

        assert word_sync_step_limit > 1

        # debug
        # print(f"{sent_lengths=}")
        # debug

        # Initialize beam.
        open_beam = self.build_new_beam(
            initial_hidden=self.rnng.get_initial_hidden(x),
            batch_size=batch_size,
            beam_size=beam_size,
            sample_size=1,  # During inference, sample_size is fixed to 1.
            stack_size=stack_size,
            max_actions=max_actions,
            start_empty=False,  # open_beam starts with one active item.
        )
        step_complete_beam = self.build_new_beam(
            initial_hidden=self.rnng.get_initial_hidden(x),
            batch_size=batch_size,
            beam_size=beam_size,
            sample_size=1,  # During inference, sample_size is fixed to 1.
            stack_size=stack_size,
            max_actions=max_actions,
            start_empty=True,  # step_complete_beam is first empty.
        )

        # batch -> time step -> beam -> beam item score
        beam_scores: list[list[list[float]]] = [[] for _ in range(batch_size)]
        beam_token_log_probs: list[list[list[float]]] = [[] for _ in range(batch_size)]

        i = 0
        for word_sync_step in range(x.size(-1)):  # eos token included.
            tmp_step_count: int = 0

            finished_word_sync_step_batches = get_finished_word_sync_step_batches(
                open_beam=open_beam,
                step_complete_beam=step_complete_beam,
                word_sync_step_limit=word_sync_step_limit,
                tmp_step_count=tmp_step_count,
            )

            # Loop until there is no not-completed items or step_complete_beam is filled with completed items.
            while not finished_word_sync_step_batches.all():
                self.step_word_sync_beam_search(
                    x=x,
                    open_beam=open_beam,
                    step_complete_beam=step_complete_beam,
                    sent_lengths=sent_lengths,
                    min_shift_size=min_shift_size,
                    finished_word_sync_step_batches=finished_word_sync_step_batches,
                    word_sync_step=word_sync_step,
                )
                # Update finished mask.
                finished_word_sync_step_batches = get_finished_word_sync_step_batches(
                    open_beam=open_beam,
                    step_complete_beam=step_complete_beam,
                    word_sync_step_limit=word_sync_step_limit,
                    tmp_step_count=tmp_step_count,
                )

                # Update counts.
                tmp_step_count += 1
                i += 1

            # Complete step_complete_beam.
            active_widths_before_move, beam_scores_before_token_prediction, token_log_probs = (
                self.complete_word_sync_step(
                    x=x,
                    open_beam=open_beam,
                    step_complete_beam=step_complete_beam,
                    sent_lengths=sent_lengths,
                    word_sync_step=word_sync_step,
                )
            )

            # Convert the scores into list.
            for batch_i, sent_len in enumerate(sent_lengths.cpu().detach()):
                # Only store the results of unfinished batches.
                if word_sync_step < sent_len:
                    active_width = active_widths_before_move[batch_i].cpu().detach()

                    # Only get the scores of active items.
                    # Note the sample_size is always 1 here.
                    cur_beam_scores: list[float] = beam_scores_before_token_prediction[
                        batch_i, 0, :active_width
                    ].tolist()
                    cur_beam_token_log_probs: list[float] = token_log_probs[batch_i, 0, :active_width].tolist()

                    # Add results.
                    beam_scores[batch_i].append(cur_beam_scores)
                    beam_token_log_probs[batch_i].append(cur_beam_token_log_probs)

        # debug
        assert step_complete_beam.active_idx_mask().any(dim=-1).all()
        # debug

        # Get the inferenced actions.

        best_beam_item_mask = step_complete_beam.get_best_item_mask()

        # Note the offset of beam.actions; the first elements of beam.actions are <PAD> (which is the seudo previous action of the first step).
        best_actions: torch.Tensor = step_complete_beam.actions[..., 1:][best_beam_item_mask].view(
            step_complete_beam.batch_size, step_complete_beam.sample_size, -1
        )
        assert best_actions.size() == (
            step_complete_beam.batch_size,
            step_complete_beam.sample_size,
            step_complete_beam.actions.size(-1) - 1,  # Note the offset.
        )

        action_lens: torch.Tensor = (best_actions != self.action_dict.padding_idx).count_nonzero(dim=-1)
        assert action_lens.size() == (step_complete_beam.batch_size, step_complete_beam.sample_size)

        assert step_complete_beam.sample_size == 1
        best_action_lists: list[list[int]] = [
            best_actions[
                batch_i,
                0,
                : action_lens[batch_i, 0],  # Note that sample dimension is assumed to be 1.
            ].tolist()
            for batch_i in range(step_complete_beam.batch_size)
        ]

        return best_action_lists, beam_scores, beam_token_log_probs

    def supervised_forward(
        self,
        x: torch.Tensor,  # (batch_size, sent_len)
        actions: torch.Tensor,  # (batch_size, action_len)
        stack_size: int,
    ) -> tuple[
        torch.Tensor,  # (batch_size,) Total loss.
        torch.Tensor,  # (batch_size,) Surface loss.
        torch.Tensor,  # (batch_size,) Structure loss.
        torch.Tensor,  # (batch_size,) Token PPL.
        torch.Tensor,  # (batch_size,) Action PPL.
        list[int],  # List containing nt_pos counts.
    ]:
        batch_size = x.size(0)
        max_action_len = actions.size(-1)
        max_sent_len = x.size(-1)

        nt_pos_count = self.count_nt_pos(actions=actions)
        assert actions.size(0) == x.size(0)

        initial_hidden = self.rnng.get_initial_hidden(x)
        stack = GeneralizedActionFixedStack(
            initial_hidden=initial_hidden,
            stack_size=stack_size,
            input_size=self.input_size,
            batch_size=batch_size,
            beam_size=1,
            sample_size=1,
        )

        hs: torch.Tensor = initial_hidden[0].new_zeros(
            batch_size,
            max_action_len,
            self.hidden_size,
        )

        for step in range(max_action_len):
            # Take the hidden state before each action.
            hs[:, step, :] = stack.hidden_head()[:, :, -1].view(batch_size, self.hidden_size)

            # Add sample/beam dimension to actions.
            self.rnng(x=x, actions=actions[:, step].view(-1, 1, 1), stack=stack)

        # Calculate masks for padding.
        # Remove sample dimension of actions when calculating pad actions.
        # Note that out of width samples in the actions are filled with <PAD> actions.
        non_pad_action_idxs = (actions != self.action_dict.padding_idx).nonzero(as_tuple=True)
        shift_action_idxs = (actions == self.action_dict.shift_idx).nonzero(as_tuple=True)

        token_non_padding_idxs = (x != self.vocab_padding_idx).nonzero(as_tuple=True)
        assert (shift_action_idxs[0] == token_non_padding_idxs[0]).all()  # Check batches.

        # Calculate MLE loss.

        # Calculate the inputs for vocab/action MLPs.
        hiddens: torch.Tensor = initial_hidden[0].new_zeros(
            batch_size,
            max_action_len,
            self.input_size,
        )
        hiddens[non_pad_action_idxs] = self.rnng.output(hs[non_pad_action_idxs])

        # Calculate loss for parsing actions.
        action_logits = torch.tensor([-float("inf")], device=x.device).repeat(
            batch_size, max_action_len, self.action_size
        )  # (batch_sizse, action_len, action_size)
        action_logits[non_pad_action_idxs] = self.action_mlp(hiddens[non_pad_action_idxs])

        # The cross entropy for ignore_idx will be 0.
        action_losses = (
            torch.nn.functional.cross_entropy(
                input=action_logits.view(batch_size * max_action_len, self.action_size),
                target=actions.view(-1),
                reduction="none",
                ignore_index=self.action_dict.padding_idx,
            )
            .view(batch_size, max_action_len)
            .sum(dim=-1)  # Sum over time steps.
        )
        assert action_losses.size() == (batch_size,)

        action_lens = (actions != self.action_dict.padding_idx).count_nonzero(dim=-1)
        action_ppl = torch.exp(action_losses / action_lens)
        assert action_ppl.size() == (batch_size,)

        # Calculate loss for token prediction.
        token_logits = torch.tensor([-float("inf")], device=x.device).repeat(
            batch_size,
            max_sent_len,
            self.vocab_size,
        )  # (batch_size, sample_size, action_size)
        token_logits[token_non_padding_idxs] = self.vocab_mlp(hiddens[shift_action_idxs])

        # The cross entropy for ignore_idx will be 0.
        token_losses = (
            torch.nn.functional.cross_entropy(
                input=token_logits.view(batch_size * max_sent_len, -1),
                target=x.view(batch_size * max_sent_len),
                reduction="none",
                ignore_index=self.vocab_padding_idx,
            )
            .view(batch_size, max_sent_len)
            .sum(dim=-1)  # Sum over time steps.
        )
        assert token_losses.size() == (batch_size,)

        num_tokens = (x != self.vocab_padding_idx).count_nonzero(dim=-1)
        token_ppl = torch.exp(token_losses / num_tokens)
        assert token_ppl.size() == (batch_size,)

        total_loss = token_losses + action_losses

        return (
            total_loss,
            token_losses,
            action_losses,
            token_ppl,
            action_ppl,
            nt_pos_count,
        )
