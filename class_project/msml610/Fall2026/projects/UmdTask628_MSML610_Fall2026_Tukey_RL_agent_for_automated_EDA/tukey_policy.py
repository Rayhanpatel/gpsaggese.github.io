"""
The Tukey policy: a tool tower that picks tool calls and a decoder tower that
emits a DAG.

Both towers have the same encoder architecture (separate weights): per-node and
per-pair MLPs, one round of message passing with mean and max pooling, and a
global summary. There are no index embeddings, so relabelling the variables
relabels the outputs. Every module works on a batch of states with the same
number of variables (leading batch dimension); the single-state methods add
and remove that dimension.

Decoder: node utilities `u` define a Plackett-Luce distribution over causal
orders; given an order, each pair `i` before `j` gets the edge i -> j with
probability `sigmoid(l_ij)`. Every emitted graph is acyclic, any DAG can be
emitted, and `log p(G, order) = log PL(order) + sum of Bernoulli terms` is
exact, so policy gradients use exactly the graph that is scored.

Tool head: a masked softmax over the legal tool calls and STOP; each candidate
is scored from the embeddings of the variables it involves. Its first layer is
factorised (a linear map of each embedding slot, then a sum), so per-candidate
inputs are never materialised.

A value head on the tool tower (`state_value`) predicts the return from a
state; it reads detached embeddings, so it never changes the policy's features,
and serves only as a baseline.

Import as:

import tukey_policy as tpolic
"""

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

import helpers.hdbg as hdbg
import tukey_features as tfeat
import tukey_tools as ttools

_LOG = logging.getLogger(__name__)

# Candidate types of the tool head.
_TYPES = (
    "ci_1",
    "ci_2",
    "regress_1",
    "regress_2",
    "lingam",
    "nongauss",
    "variance",
    "stop",
)
_TYPE_INDEX = {name: k for k, name in enumerate(_TYPES)}

Batch = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def _mlp(n_in: int, n_hidden: int, n_out: int) -> torch.nn.Module:
    """
    Build a two-layer perceptron.

    :param n_in: input width
    :param n_hidden: hidden width
    :param n_out: output width
    :return: module
    """
    mlp = torch.nn.Sequential(
        torch.nn.Linear(n_in, n_hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(n_hidden, n_out),
    )
    return mlp


def stack_observations(
    observations: Sequence[tfeat.Observation],
    *,
    dtype: torch.dtype = torch.float32,
) -> Batch:
    """
    Stack observations with the same number of variables into tensors.

    :param observations: observations
    :param dtype: tensor type
    :return: node (B, d, F_node), pair (B, d, d, F_pair), global (B, F_glob)
    """
    node = torch.as_tensor(np.stack([o.node for o in observations]), dtype=dtype)
    pair = torch.as_tensor(np.stack([o.pair for o in observations]), dtype=dtype)
    glob = torch.as_tensor(np.stack([o.glob for o in observations]), dtype=dtype)
    return node, pair, glob


# #############################################################################
# Encoder
# #############################################################################


class Encoder(torch.nn.Module):
    """
    Embed nodes, ordered pairs and the global state of a batch of states.
    """

    def __init__(self, *, hidden: int = 64) -> None:
        """
        Build the encoder.

        :param hidden: embedding width
        """
        super().__init__()
        n_node = len(tfeat.NODE_FEATURES)
        n_pair = len(tfeat.PAIR_FEATURES)
        n_glob = len(tfeat.GLOBAL_FEATURES)
        self.hidden = hidden
        self.node_in = _mlp(n_node + n_glob, hidden, hidden)
        self.pair_in = _mlp(n_pair + n_glob, hidden, hidden)
        self.node_update = _mlp(5 * hidden, hidden, hidden)
        self.pair_update = _mlp(3 * hidden, hidden, hidden)
        self.glob_out = _mlp(n_glob + 2 * hidden, hidden, hidden)

    def forward(
        self, node: torch.Tensor, pair: torch.Tensor, glob: torch.Tensor
    ) -> Batch:
        """
        Embed a batch of states.

        :param node: node features (B, d, F_node)
        :param pair: pair features (B, d, d, F_pair)
        :param glob: global features (B, F_glob)
        :return: node (B, d, H), pair (B, d, d, H) and global (B, H) embeddings
        """
        n_batch, n_nodes = node.shape[0], node.shape[1]
        glob_node = glob[:, None, :].expand(n_batch, n_nodes, -1)
        glob_pair = glob[:, None, None, :].expand(n_batch, n_nodes, n_nodes, -1)
        node_h = torch.relu(self.node_in(torch.cat([node, glob_node], dim=-1)))
        pair_h = torch.relu(self.pair_in(torch.cat([pair, glob_pair], dim=-1)))
        # Pool over the other variables, excluding the diagonal.
        off_diag = ~torch.eye(n_nodes, dtype=torch.bool)
        mask = off_diag[None, :, :, None].to(pair_h.dtype)
        n_others = max(1, n_nodes - 1)
        out_mean = (pair_h * mask).sum(dim=2) / n_others
        in_mean = (pair_h * mask).sum(dim=1) / n_others
        if n_nodes > 1:
            very_negative = torch.finfo(pair_h.dtype).min
            masked = pair_h.masked_fill(
                ~off_diag[None, :, :, None], very_negative
            )
            out_max = masked.max(dim=2).values
            in_max = masked.max(dim=1).values
        else:
            out_max = torch.zeros_like(node_h)
            in_max = torch.zeros_like(node_h)
        node_h = torch.relu(
            self.node_update(
                torch.cat([node_h, out_mean, out_max, in_mean, in_max], dim=-1)
            )
        )
        node_i = node_h[:, :, None, :].expand(n_batch, n_nodes, n_nodes, -1)
        node_j = node_h[:, None, :, :].expand(n_batch, n_nodes, n_nodes, -1)
        pair_h = torch.relu(
            self.pair_update(torch.cat([pair_h, node_i, node_j], dim=-1))
        )
        n_pairs = max(1, n_nodes * (n_nodes - 1))
        pair_mean = (pair_h * mask).sum(dim=(1, 2)) / n_pairs
        glob_h = torch.relu(
            self.glob_out(
                torch.cat([glob, node_h.mean(dim=1), pair_mean], dim=-1)
            )
        )
        return node_h, pair_h, glob_h


# #############################################################################
# Decoder: Plackett-Luce order, then Bernoulli edges along it
# #############################################################################


def _precedence(orders: torch.Tensor) -> torch.Tensor:
    """
    Compute "i comes before j" for each order.

    :param orders: node orders (..., d), first node first
    :return: boolean tensor (..., d, d), True where i precedes j
    """
    n_nodes = orders.shape[-1]
    ranks = torch.arange(n_nodes).expand(orders.shape).contiguous()
    position = torch.empty_like(orders).scatter_(-1, orders, ranks)
    before = position.unsqueeze(-1) < position.unsqueeze(-2)
    return before


def order_edge_log_prob(
    u: torch.Tensor,
    logits: torch.Tensor,
    orders: torch.Tensor,
    B: torch.Tensor,
) -> torch.Tensor:
    """
    Compute `log p(G, order)` of sampled graphs.

    Leading dimensions are batch dimensions: `u` (..., d) and `logits`
    (..., d, d) are shared by the M samples in `orders` (..., M, d) and `B`
    (..., M, d, d).

    :param u: node utilities
    :param logits: edge logits
    :param orders: node orders, first node first
    :param B: graphs, with edges only along their order
    :return: log-probabilities (..., M)
    """
    u_rep = u.unsqueeze(-2).expand(orders.shape)
    u_ordered = torch.gather(u_rep, -1, orders)
    # log PL = sum_k [u_(k) - logsumexp(u_(k..d))].
    tail_lse = torch.logcumsumexp(u_ordered.flip(-1), dim=-1).flip(-1)
    log_p_order = (u_ordered - tail_lse).sum(dim=-1)
    before = _precedence(orders).to(logits.dtype)
    B = B.to(logits.dtype)
    logits = logits.unsqueeze(-3)
    log_bern = B * torch.nn.functional.logsigmoid(logits) + (
        1 - B
    ) * torch.nn.functional.logsigmoid(-logits)
    log_p_edges = (log_bern * before).sum(dim=(-1, -2))
    return log_p_order + log_p_edges


def sample_graphs(
    u: torch.Tensor,
    logits: torch.Tensor,
    n_samples: int,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample orders (Gumbel-top-k) and edges along them.

    :param u: node utilities (..., d)
    :param logits: edge logits (..., d, d)
    :param n_samples: graphs per state (M)
    :param generator: random generator
    :return: orders (..., M, d) and graphs (..., M, d, d) as integer tensors
    """
    with torch.no_grad():
        u_rep = u.unsqueeze(-2).expand(*u.shape[:-1], n_samples, u.shape[-1])
        uniform = torch.rand(u_rep.shape, generator=generator, dtype=u.dtype)
        gumbel = -torch.log(-torch.log(uniform.clamp_min(1e-12)))
        orders = torch.argsort(u_rep + gumbel, dim=-1, descending=True)
        before = _precedence(orders)
        probs = torch.sigmoid(logits).unsqueeze(-3)
        coins = torch.rand(before.shape, generator=generator, dtype=u.dtype)
        B = ((coins < probs) & before).long()
    return orders, B


def greedy_graph(u: torch.Tensor, logits: torch.Tensor) -> np.ndarray:
    """
    Decode the most likely order and the edges with probability above 1/2.

    :param u: node utilities (d,)
    :param logits: edge logits (d, d)
    :return: DAG adjacency
    """
    with torch.no_grad():
        order = torch.argsort(u, descending=True)
        before = _precedence(order)
        B = ((logits > 0) & before).long()
    return B.numpy()


def topological_order(B: np.ndarray, u: np.ndarray) -> List[int]:
    """
    Order a DAG's nodes, breaking ties by the larger utility.

    Used to give an expert's DAG the order the decoder would most likely use.

    :param B: DAG adjacency
    :param u: node utilities
    :return: order with every parent before its children
    """
    n_nodes = B.shape[0]
    in_degree = B.sum(axis=0).astype(int)
    remaining = set(range(n_nodes))
    order = []
    while remaining:
        ready = [k for k in remaining if in_degree[k] == 0]
        hdbg.dassert_lt(0, len(ready), "Graph has a cycle")
        node = max(ready, key=lambda k: (u[k], -k))
        order.append(node)
        remaining.remove(node)
        for child in np.flatnonzero(B[node]):
            in_degree[child] -= 1
    return order


# #############################################################################
# Tool head
# #############################################################################


def _candidate_index(
    actions: List[ttools.Action], n_nodes: int
) -> Dict[str, torch.Tensor]:
    """
    Precompute which embeddings each candidate action reads.

    Each candidate has a pair slot (up to 2 ordered pairs), three node slots
    (node A, node B, conditioning set; up to 2 nodes each) and a context slot
    (up to 8 ordered pairs linking a CI test's conditioning variables to its
    pair, or a regression's two predictors), averaged with the given weights;
    index `n_nodes` (nodes) or `n_nodes**2` (pairs) points at a zero row.

    :param actions: candidate actions (tool calls, then STOP)
    :param n_nodes: number of variables
    :return: `pair_idx`, `node_idx` (index tensors) and their weights, `types`
    """
    null_node = n_nodes
    null_pair = n_nodes * n_nodes
    n_cand = len(actions)
    pair_idx = np.full((n_cand, 2), null_pair)
    node_idx = np.full((n_cand, 3, 2), null_node)
    context_idx = np.full((n_cand, 8), null_pair)
    pair_w = np.zeros((n_cand, 2))
    node_w = np.zeros((n_cand, 3, 2))
    context_w = np.zeros((n_cand, 8))
    types = np.zeros(n_cand, dtype=int)
    for c, action in enumerate(actions):
        args = action.args
        if action.kind == ttools.CI_TEST:
            i, j, cond = args[0], args[1], args[2:]
            pair_idx[c] = (i * n_nodes + j, j * n_nodes + i)
            pair_w[c] = 0.5
            node_idx[c, 0] = (i, j)
            node_w[c, 0] = 0.5
            node_idx[c, 2, : len(cond)] = cond
            node_w[c, 2, : len(cond)] = 1.0 / len(cond)
            # Context: how each conditioning variable relates to i and j.
            links = []
            for k in cond:
                links += [i * n_nodes + k, k * n_nodes + i]
                links += [j * n_nodes + k, k * n_nodes + j]
            context_idx[c, : len(links)] = links
            context_w[c, : len(links)] = 1.0 / len(links)
            types[c] = _TYPE_INDEX[f"ci_{len(cond)}"]
        elif action.kind == ttools.REGRESS:
            target, preds = args[0], args[1:]
            pair_idx[c, : len(preds)] = [p * n_nodes + target for p in preds]
            pair_w[c, : len(preds)] = 1.0 / len(preds)
            node_idx[c, 0, 0] = target
            node_w[c, 0, 0] = 1.0
            node_idx[c, 1, : len(preds)] = preds
            node_w[c, 1, : len(preds)] = 1.0 / len(preds)
            if len(preds) == 2:
                # Context: how the two predictors relate to each other.
                p_1, p_2 = preds
                context_idx[c, :2] = (p_1 * n_nodes + p_2, p_2 * n_nodes + p_1)
                context_w[c, :2] = 0.5
            types[c] = _TYPE_INDEX[f"regress_{len(preds)}"]
        elif action.kind == ttools.LINGAM_DIR:
            i, j = args
            pair_idx[c] = (i * n_nodes + j, j * n_nodes + i)
            pair_w[c] = 0.5
            node_idx[c, 0] = (i, j)
            node_w[c, 0] = 0.5
            types[c] = _TYPE_INDEX["lingam"]
        elif action.kind in (ttools.NONGAUSS, ttools.VARIANCE):
            node_idx[c, 0, 0] = args[0]
            node_w[c, 0, 0] = 1.0
            types[c] = _TYPE_INDEX[action.kind]
        elif action.kind == ttools.STOP:
            types[c] = _TYPE_INDEX["stop"]
        else:
            raise ValueError(f"Unknown action '{action.kind}'")
    index = {
        "pair_idx": torch.as_tensor(pair_idx),
        "pair_w": torch.as_tensor(pair_w),
        "node_idx": torch.as_tensor(node_idx),
        "node_w": torch.as_tensor(node_w),
        "context_idx": torch.as_tensor(context_idx),
        "context_w": torch.as_tensor(context_w),
        "types": torch.as_tensor(types),
    }
    return index


class ToolHead(torch.nn.Module):
    """
    Score every candidate tool call and STOP for a batch of states.
    """

    def __init__(self, *, hidden: int = 64) -> None:
        """
        Build the head.

        :param hidden: embedding width
        """
        super().__init__()
        # Factorised first layer: one linear map per slot, then a sum.
        self.proj_pair = torch.nn.Linear(hidden, hidden, bias=False)
        self.proj_context = torch.nn.Linear(hidden, hidden, bias=False)
        self.proj_nodes = torch.nn.ModuleList(
            [torch.nn.Linear(hidden, hidden, bias=False) for _ in range(3)]
        )
        self.proj_glob = torch.nn.Linear(hidden, hidden)
        self.type_embedding = torch.nn.Embedding(len(_TYPES), hidden)
        self.out = torch.nn.Linear(hidden, 1)

    def forward(
        self,
        node_h: torch.Tensor,
        pair_h: torch.Tensor,
        glob_h: torch.Tensor,
        index: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute one logit per candidate and state.

        :param node_h: node embeddings (B, d, H)
        :param pair_h: pair embeddings (B, d, d, H)
        :param glob_h: global embeddings (B, H)
        :param index: output of `_candidate_index()`
        :return: logits (B, n_candidates)
        """
        n_batch, n_nodes, hidden = node_h.shape
        dtype = node_h.dtype
        zeros = node_h.new_zeros(n_batch, 1, hidden)
        node_table = torch.cat([node_h, zeros], dim=1)
        pair_table = torch.cat(
            [pair_h.reshape(n_batch, n_nodes * n_nodes, hidden), zeros], dim=1
        )
        pair_proj = self.proj_pair(pair_table)[:, index["pair_idx"]]
        pair_w = index["pair_w"].to(dtype)[None, :, :, None]
        hidden_sum = (pair_proj * pair_w).sum(dim=2)
        context_proj = self.proj_context(pair_table)[:, index["context_idx"]]
        context_w = index["context_w"].to(dtype)[None, :, :, None]
        hidden_sum = hidden_sum + (context_proj * context_w).sum(dim=2)
        node_w = index["node_w"].to(dtype)
        for slot, proj in enumerate(self.proj_nodes):
            slot_proj = proj(node_table)[:, index["node_idx"][:, slot]]
            hidden_sum = hidden_sum + (
                slot_proj * node_w[None, :, slot, :, None]
            ).sum(dim=2)
        hidden_sum = hidden_sum + self.proj_glob(glob_h)[:, None, :]
        hidden_sum = hidden_sum + self.type_embedding(index["types"])[None]
        logits = self.out(torch.relu(hidden_sum)).squeeze(-1)
        return logits


# #############################################################################
# TukeyPolicy
# #############################################################################


class TukeyPolicy(torch.nn.Module):
    """
    Tool tower and decoder tower with separate weights, plus a value head.
    """

    def __init__(self, *, hidden: int = 64) -> None:
        """
        Build the policy.

        :param hidden: embedding width
        """
        super().__init__()
        self.hidden = hidden
        self.tool_encoder = Encoder(hidden=hidden)
        self.tool_head = ToolHead(hidden=hidden)
        self.decoder_encoder = Encoder(hidden=hidden)
        self.node_score = _mlp(2 * hidden, hidden, 1)
        self.edge_score = _mlp(2 * hidden, hidden, 1)
        self.value_head = _mlp(2 * hidden, hidden, 1)
        self._candidates: Dict[
            int, Tuple[List[ttools.Action], Dict[str, torch.Tensor]]
        ] = {}

    @property
    def dtype(self) -> torch.dtype:
        """
        Floating-point type of the parameters.
        """
        return next(self.parameters()).dtype

    def candidates(
        self, n_nodes: int
    ) -> Tuple[List[ttools.Action], Dict[str, torch.Tensor]]:
        """
        Return all candidate actions for d variables (tool calls, then STOP).

        :param n_nodes: number of variables
        :return: actions and their embedding index
        """
        if n_nodes not in self._candidates:
            actions = ttools.enumerate_tool_actions(n_nodes)
            actions.append(ttools.Action(ttools.STOP))
            index = _candidate_index(actions, n_nodes)
            self._candidates[n_nodes] = (actions, index)
        return self._candidates[n_nodes]

    def tool_log_probs_batch(
        self, batch: Batch, legal: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute log-probabilities of every candidate and the state values.

        :param batch: stacked observations (`stack_observations()`)
        :param legal: boolean mask (B, n_candidates)
        :return: log-probabilities (B, n_candidates), -inf when illegal, and
            state values (B,) from the value head (detached features)
        """
        node_h, pair_h, glob_h = self.tool_encoder(*batch)
        _, index = self.candidates(node_h.shape[1])
        logits = self.tool_head(node_h, pair_h, glob_h, index)
        legal_t = torch.as_tensor(legal, dtype=torch.bool)
        logits = logits.masked_fill(~legal_t, float("-inf"))
        log_probs = torch.log_softmax(logits, dim=-1)
        summary = torch.cat([glob_h, node_h.mean(dim=1)], dim=-1).detach()
        values = self.value_head(summary).squeeze(-1)
        return log_probs, values

    def decoder_params_batch(
        self, batch: Batch
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute order utilities and edge logits for a batch of states.

        :param batch: stacked observations
        :return: `u` (B, d) and `logits` (B, d, d)
        """
        node_h, pair_h, glob_h = self.decoder_encoder(*batch)
        n_batch, n_nodes = node_h.shape[0], node_h.shape[1]
        glob_node = glob_h[:, None, :].expand(n_batch, n_nodes, -1)
        glob_pair = glob_h[:, None, None, :].expand(
            n_batch, n_nodes, n_nodes, -1
        )
        u = self.node_score(torch.cat([node_h, glob_node], dim=-1)).squeeze(-1)
        logits = self.edge_score(torch.cat([pair_h, glob_pair], dim=-1)).squeeze(
            -1
        )
        return u, logits

    def tool_log_probs(
        self, obs: tfeat.Observation, legal: np.ndarray
    ) -> torch.Tensor:
        """
        Compute log-probabilities of every candidate for one state.

        :param obs: observation
        :param legal: boolean mask over `candidates(d)`
        :return: log-probabilities (n_candidates,), -inf when illegal
        """
        batch = stack_observations([obs], dtype=self.dtype)
        log_probs, _ = self.tool_log_probs_batch(batch, legal[None])
        return log_probs[0]

    def decoder_params(
        self, obs: tfeat.Observation
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the order utilities and edge logits for one state.

        :param obs: observation
        :return: `u` (d,) and `logits` (d, d)
        """
        batch = stack_observations([obs], dtype=self.dtype)
        u, logits = self.decoder_params_batch(batch)
        return u[0], logits[0]


def legal_mask(
    actions: List[ttools.Action], evidence: ttools.Evidence
) -> np.ndarray:
    """
    Mark which candidates are legal: unused tool calls within budget, and STOP.

    :param actions: candidates from `TukeyPolicy.candidates()`
    :param evidence: evidence so far
    :return: boolean mask
    """
    has_budget = evidence.budget_left > 0
    mask = np.array(
        [
            a.kind == ttools.STOP or (has_budget and a not in evidence.results)
            for a in actions
        ]
    )
    return mask


# #############################################################################
# PolicyAgent
# #############################################################################


class PolicyAgent:
    """
    Play a `TukeyPolicy` through the scripted-agent interface.
    """

    def __init__(
        self,
        policy: TukeyPolicy,
        *,
        greedy: bool,
        seed: int = 0,
        decoder_policy: Optional[TukeyPolicy] = None,
    ) -> None:
        """
        Wrap a policy.

        :param policy: policy whose tool tower picks calls
        :param greedy: take the most likely action and graph
        :param seed: seed of the sampling generator
        :param decoder_policy: policy whose decoder emits the graph
            - Default: None means `policy`
        """
        self.policy = policy
        self.decoder_policy = decoder_policy or policy
        self.greedy = greedy
        self.generator = torch.Generator().manual_seed(seed)

    def next_action(self, evidence: ttools.Evidence) -> ttools.Action:
        """
        Pick the next tool call or STOP.

        :param evidence: evidence so far
        :return: action
        """
        actions, _ = self.policy.candidates(evidence.n_nodes)
        legal = legal_mask(actions, evidence)
        with torch.no_grad():
            log_probs = self.policy.tool_log_probs(
                tfeat.featurize(evidence), legal
            )
        if self.greedy:
            choice = int(torch.argmax(log_probs))
        else:
            choice = int(
                torch.multinomial(log_probs.exp(), 1, generator=self.generator)
            )
        return actions[choice]

    def emit_graph(self, evidence: ttools.Evidence) -> List[Tuple[int, int]]:
        """
        Decode a DAG from the evidence.

        :param evidence: evidence at the end of the episode
        :return: edges `[(parent, child), ...]`
        """
        with torch.no_grad():
            u, logits = self.decoder_policy.decoder_params(
                tfeat.featurize(evidence)
            )
        if self.greedy:
            B = greedy_graph(u, logits)
        else:
            _, B_t = sample_graphs(u, logits, 1, self.generator)
            B = B_t[0].numpy()
        edges = [(int(i), int(j)) for i, j in np.argwhere(B == 1)]
        return edges
