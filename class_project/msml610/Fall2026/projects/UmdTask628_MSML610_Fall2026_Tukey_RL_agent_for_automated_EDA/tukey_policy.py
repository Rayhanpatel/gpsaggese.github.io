"""
The Tukey policy: a tool tower that picks tool calls and a decoder tower that
emits a DAG.

Both towers have the same encoder architecture (separate weights): per-node and
per-pair MLPs, one round of message passing with mean and max pooling, and a
global summary. There are no index embeddings, so relabelling the variables
relabels the outputs.

Decoder: node utilities `u` define a Plackett-Luce distribution over causal
orders; given an order, each pair `i` before `j` gets the edge i -> j with
probability `sigmoid(l_ij)`. Every emitted graph is acyclic, any DAG can be
emitted, and `log p(G, order) = log PL(order) + sum of Bernoulli terms` is
exact, so policy gradients use exactly the graph that is scored.

Tool head: a masked softmax over the legal tool calls and STOP; each candidate
is scored from the embeddings of the variables it involves.

Import as:

import tukey_policy as tpolic
"""

import logging
from typing import Dict, List, Optional, Tuple

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
# Embedding slots of a candidate: pair, node A, node B, conditioning set.
_N_SLOTS = 4


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


def obs_to_tensors(
    obs: tfeat.Observation, *, dtype: torch.dtype = torch.float32
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert an observation to tensors.

    :param obs: observation
    :param dtype: tensor type
    :return: node, pair and global feature tensors
    """
    node = torch.as_tensor(obs.node, dtype=dtype)
    pair = torch.as_tensor(obs.pair, dtype=dtype)
    glob = torch.as_tensor(obs.glob, dtype=dtype)
    return node, pair, glob


# #############################################################################
# Encoder
# #############################################################################


class Encoder(torch.nn.Module):
    """
    Embed nodes, ordered pairs and the global state.
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Embed one state.

        :param node: node features, shape (d, F_node)
        :param pair: pair features, shape (d, d, F_pair)
        :param glob: global features, shape (F_glob,)
        :return: node (d, H), pair (d, d, H) and global (H,) embeddings
        """
        n_nodes = node.shape[0]
        node_h = torch.relu(
            self.node_in(torch.cat([node, glob.expand(n_nodes, -1)], dim=-1))
        )
        pair_h = torch.relu(
            self.pair_in(
                torch.cat([pair, glob.expand(n_nodes, n_nodes, -1)], dim=-1)
            )
        )
        # Pool over the other variables, excluding the diagonal.
        off_diag = ~torch.eye(n_nodes, dtype=torch.bool)
        mask = off_diag.unsqueeze(-1).to(pair_h.dtype)
        n_others = max(1, n_nodes - 1)
        out_mean = (pair_h * mask).sum(dim=1) / n_others
        in_mean = (pair_h * mask).sum(dim=0) / n_others
        very_negative = torch.finfo(pair_h.dtype).min
        masked = pair_h.masked_fill(~off_diag.unsqueeze(-1), very_negative)
        if n_nodes > 1:
            out_max = masked.max(dim=1).values
            in_max = masked.max(dim=0).values
        else:
            out_max = torch.zeros_like(node_h)
            in_max = torch.zeros_like(node_h)
        node_h = torch.relu(
            self.node_update(
                torch.cat([node_h, out_mean, out_max, in_mean, in_max], dim=-1)
            )
        )
        pair_h = torch.relu(
            self.pair_update(
                torch.cat(
                    [
                        pair_h,
                        node_h.unsqueeze(1).expand(n_nodes, n_nodes, -1),
                        node_h.unsqueeze(0).expand(n_nodes, n_nodes, -1),
                    ],
                    dim=-1,
                )
            )
        )
        pair_mean = (pair_h * mask).sum(dim=(0, 1)) / max(
            1, n_nodes * (n_nodes - 1)
        )
        glob_h = torch.relu(
            self.glob_out(
                torch.cat([glob, node_h.mean(dim=0), pair_mean], dim=-1)
            )
        )
        return node_h, pair_h, glob_h


# #############################################################################
# Decoder: Plackett-Luce order, then Bernoulli edges along it
# #############################################################################


def _precedence(orders: torch.Tensor) -> torch.Tensor:
    """
    Compute "i comes before j" for each sampled order.

    :param orders: node orders, shape (M, d)
    :return: boolean tensor (M, d, d), True where i precedes j
    """
    n_samples, n_nodes = orders.shape
    position = torch.empty_like(orders)
    position.scatter_(
        1, orders, torch.arange(n_nodes).expand(n_samples, n_nodes).contiguous()
    )
    before = position.unsqueeze(2) < position.unsqueeze(1)
    return before


def order_edge_log_prob(
    u: torch.Tensor,
    logits: torch.Tensor,
    orders: torch.Tensor,
    B: torch.Tensor,
) -> torch.Tensor:
    """
    Compute `log p(G, order)` of sampled graphs.

    :param u: node utilities, shape (d,)
    :param logits: edge logits, shape (d, d)
    :param orders: node orders, shape (M, d), first node first
    :param B: graphs, shape (M, d, d), only edges along the order
    :return: log-probabilities, shape (M,)
    """
    u_ordered = u[orders]
    # log PL = sum_k [u_(k) - logsumexp(u_(k..d))].
    tail_lse = torch.logcumsumexp(u_ordered.flip(-1), dim=-1).flip(-1)
    log_p_order = (u_ordered - tail_lse).sum(dim=-1)
    before = _precedence(orders).to(logits.dtype)
    B = B.to(logits.dtype)
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

    :param u: node utilities, shape (d,)
    :param logits: edge logits, shape (d, d)
    :param n_samples: number of graphs
    :param generator: random generator
    :return: orders (M, d) and graphs (M, d, d) as integer tensors
    """
    n_nodes = u.shape[0]
    with torch.no_grad():
        uniform = torch.rand(
            n_samples, n_nodes, generator=generator, dtype=u.dtype
        )
        gumbel = -torch.log(-torch.log(uniform.clamp_min(1e-12)))
        orders = torch.argsort(u.unsqueeze(0) + gumbel, dim=-1, descending=True)
        before = _precedence(orders)
        coins = torch.rand(
            n_samples, n_nodes, n_nodes, generator=generator, dtype=u.dtype
        )
        B = ((coins < torch.sigmoid(logits)) & before).long()
    return orders, B


def greedy_graph(u: torch.Tensor, logits: torch.Tensor) -> np.ndarray:
    """
    Decode the most likely order and the edges with probability above 1/2.

    :param u: node utilities
    :param logits: edge logits
    :return: DAG adjacency
    """
    with torch.no_grad():
        order = torch.argsort(u, descending=True).unsqueeze(0)
        before = _precedence(order)[0]
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

    Each candidate has 4 slots (pair, node A, node B, conditioning set) of up to
    2 indices each, averaged with the given weights; index `n_nodes` (nodes) or
    `n_nodes**2` (pairs) points at a zero row.

    :param actions: candidate actions (tool calls, then STOP)
    :param n_nodes: number of variables
    :return: `pair_idx`, `node_idx` (index tensors) and their weights, `types`
    """
    null_node = n_nodes
    null_pair = n_nodes * n_nodes
    n_cand = len(actions)
    pair_idx = np.full((n_cand, 2), null_pair)
    node_idx = np.full((n_cand, 3, 2), null_node)
    pair_w = np.zeros((n_cand, 2))
    node_w = np.zeros((n_cand, 3, 2))
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
            types[c] = _TYPE_INDEX[f"ci_{len(cond)}"]
        elif action.kind == ttools.REGRESS:
            target, preds = args[0], args[1:]
            pair_idx[c, : len(preds)] = [p * n_nodes + target for p in preds]
            pair_w[c, : len(preds)] = 1.0 / len(preds)
            node_idx[c, 0, 0] = target
            node_w[c, 0, 0] = 1.0
            node_idx[c, 1, : len(preds)] = preds
            node_w[c, 1, : len(preds)] = 1.0 / len(preds)
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
        "types": torch.as_tensor(types),
    }
    return index


class ToolHead(torch.nn.Module):
    """
    Score every candidate tool call and STOP.
    """

    def __init__(self, *, hidden: int = 64) -> None:
        """
        Build the head.

        :param hidden: embedding width
        """
        super().__init__()
        self.score = _mlp((_N_SLOTS + 1) * hidden + len(_TYPES), hidden, 1)

    def forward(
        self,
        node_h: torch.Tensor,
        pair_h: torch.Tensor,
        glob_h: torch.Tensor,
        index: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute one logit per candidate.

        :param node_h: node embeddings (d, H)
        :param pair_h: pair embeddings (d, d, H)
        :param glob_h: global embedding (H,)
        :param index: output of `_candidate_index()`
        :return: logits, shape (n_candidates,)
        """
        hidden = node_h.shape[-1]
        dtype = node_h.dtype
        node_table = torch.cat([node_h, node_h.new_zeros(1, hidden)])
        pair_table = torch.cat(
            [pair_h.reshape(-1, hidden), pair_h.new_zeros(1, hidden)]
        )
        pair_emb = (
            pair_table[index["pair_idx"]]
            * index["pair_w"].to(dtype).unsqueeze(-1)
        ).sum(1)
        node_emb = (
            node_table[index["node_idx"]]
            * index["node_w"].to(dtype).unsqueeze(-1)
        ).sum(2)
        n_cand = pair_emb.shape[0]
        one_hot = torch.nn.functional.one_hot(index["types"], len(_TYPES)).to(
            dtype
        )
        features = torch.cat(
            [
                pair_emb,
                node_emb.reshape(n_cand, -1),
                glob_h.expand(n_cand, -1),
                one_hot,
            ],
            dim=-1,
        )
        logits = self.score(features).squeeze(-1)
        return logits


# #############################################################################
# TukeyPolicy
# #############################################################################


class TukeyPolicy(torch.nn.Module):
    """
    Tool tower and decoder tower with separate weights.
    """

    def __init__(self, *, hidden: int = 64) -> None:
        """
        Build the policy.

        :param hidden: embedding width
        """
        super().__init__()
        self.tool_encoder = Encoder(hidden=hidden)
        self.tool_head = ToolHead(hidden=hidden)
        self.decoder_encoder = Encoder(hidden=hidden)
        self.node_score = _mlp(2 * hidden, hidden, 1)
        self.edge_score = _mlp(2 * hidden, hidden, 1)
        self._candidates: Dict[
            int, Tuple[List[ttools.Action], Dict[str, torch.Tensor]]
        ] = {}

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
            self._candidates[n_nodes] = (
                actions,
                _candidate_index(actions, n_nodes),
            )
        return self._candidates[n_nodes]

    def tool_log_probs(
        self, obs: tfeat.Observation, legal: np.ndarray
    ) -> torch.Tensor:
        """
        Compute log-probabilities of every candidate, -inf when illegal.

        :param obs: observation
        :param legal: boolean mask over `candidates(d)`
        :return: log-probabilities, shape (n_candidates,)
        """
        dtype = next(self.parameters()).dtype
        node_h, pair_h, glob_h = self.tool_encoder(
            *obs_to_tensors(obs, dtype=dtype)
        )
        _, index = self.candidates(obs.node.shape[0])
        logits = self.tool_head(node_h, pair_h, glob_h, index)
        legal_t = torch.as_tensor(legal, dtype=torch.bool)
        logits = logits.masked_fill(~legal_t, float("-inf"))
        log_probs = torch.log_softmax(logits, dim=-1)
        return log_probs

    def decoder_params(
        self, obs: tfeat.Observation
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the order utilities and edge logits.

        :param obs: observation
        :return: `u` (d,) and `logits` (d, d)
        """
        dtype = next(self.parameters()).dtype
        node_h, pair_h, glob_h = self.decoder_encoder(
            *obs_to_tensors(obs, dtype=dtype)
        )
        n_nodes = node_h.shape[0]
        u = self.node_score(
            torch.cat([node_h, glob_h.expand(n_nodes, -1)], dim=-1)
        ).squeeze(-1)
        logits = self.edge_score(
            torch.cat([pair_h, glob_h.expand(n_nodes, n_nodes, -1)], dim=-1)
        ).squeeze(-1)
        return u, logits


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
