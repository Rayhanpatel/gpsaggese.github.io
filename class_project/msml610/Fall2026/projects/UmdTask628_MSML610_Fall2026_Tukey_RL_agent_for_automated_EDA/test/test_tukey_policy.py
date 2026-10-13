"""
Test the policy: acyclic samples, exact probabilities, equivariance, masks.

Import as:

import test.test_tukey_policy as ttetupol
"""

import itertools
import logging
from typing import List, Tuple

import numpy as np
import torch

import helpers.hunit_test as hunitest
import tukey_env as tenv
import tukey_features as tfeat
import tukey_policy as tpolic
import tukey_scoring as tscore
import tukey_tools as ttools

_LOG = logging.getLogger(__name__)


def _observation(
    n_nodes: int, *, seed: int, n_calls: int = 30
) -> Tuple[tfeat.Observation, ttools.Evidence]:
    """
    Build an observation after random tool calls on a sampled instance.

    :param n_nodes: number of variables
    :param seed: instance and call seed
    :param n_calls: number of random tool calls
    :return: observation and its evidence
    """
    cfg = tenv.GraphConfig(family="mixed", n_nodes=n_nodes, n_edges=n_nodes)
    x = tenv.sample_instance(cfg, seed).d_train.to_numpy()
    evidence = ttools.Evidence(ttools.ToolCache(x), budget=n_calls + 10)
    actions = ttools.enumerate_tool_actions(n_nodes)
    rng = np.random.default_rng(seed)
    for k in rng.choice(
        len(actions), size=min(n_calls, len(actions)), replace=False
    ):
        evidence.record(actions[k])
    return tfeat.featurize(evidence), evidence


def _all_outcomes(n_nodes: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    List every (order, edges-along-the-order) outcome of the decoder.

    :param n_nodes: number of variables
    :return: orders (N, d) and graphs (N, d, d)
    """
    orders: List[List[int]] = []
    graphs: List[np.ndarray] = []
    n_pairs = n_nodes * (n_nodes - 1) // 2
    for order in itertools.permutations(range(n_nodes)):
        pairs = [
            (order[a], order[b])
            for a, b in itertools.combinations(range(n_nodes), 2)
        ]
        for bits in itertools.product([0, 1], repeat=n_pairs):
            B = np.zeros((n_nodes, n_nodes), dtype=int)
            for (i, j), bit in zip(pairs, bits):
                B[i, j] = bit
            orders.append(list(order))
            graphs.append(B)
    return torch.as_tensor(orders), torch.as_tensor(np.array(graphs))


class Test_decoder(hunitest.TestCase):
    """
    Check the order-then-edges decoder.
    """

    def test1(self) -> None:
        """
        Every sampled graph and the greedy graph are acyclic.
        """
        torch.manual_seed(0)
        policy = tpolic.TukeyPolicy(hidden=16)
        obs, _ = _observation(8, seed=1)
        u, logits = policy.decoder_params(obs)
        generator = torch.Generator().manual_seed(0)
        _, B = tpolic.sample_graphs(u, logits, 1000, generator)
        n_cyclic = sum(not tscore.is_dag(B[m].numpy()) for m in range(1000))
        self.assertEqual(n_cyclic, 0)
        self.assertTrue(tscore.is_dag(tpolic.greedy_graph(u, logits)))

    def test2(self) -> None:
        """
        Probabilities of all 48 outcomes for 3 variables sum to 1.
        """
        torch.manual_seed(1)
        policy = tpolic.TukeyPolicy(hidden=16).double()
        obs, _ = _observation(3, seed=2, n_calls=5)
        u, logits = policy.decoder_params(obs)
        orders, B = _all_outcomes(3)
        self.assertEqual(orders.shape[0], 48)
        log_probs = tpolic.order_edge_log_prob(u, logits, orders, B)
        total = float(torch.logsumexp(log_probs, dim=0).exp())
        self.assertAlmostEqual(total, 1.0, places=12)

    def test3(self) -> None:
        """
        Sampling frequencies match the exact probabilities (100,000 draws).
        """
        torch.manual_seed(2)
        policy = tpolic.TukeyPolicy(hidden=16).double()
        obs, _ = _observation(3, seed=3, n_calls=5)
        u, logits = policy.decoder_params(obs)
        orders, B = _all_outcomes(3)
        exact = tpolic.order_edge_log_prob(u, logits, orders, B).exp().detach()
        generator = torch.Generator().manual_seed(3)
        n_draws = 100_000
        orders_s, B_s = tpolic.sample_graphs(u, logits, n_draws, generator)
        keys = {
            (tuple(o.tolist()), b.numpy().tobytes()): k
            for k, (o, b) in enumerate(zip(orders, B))
        }
        counts = np.zeros(len(keys))
        for o, b in zip(orders_s, B_s):
            counts[keys[(tuple(o.tolist()), b.numpy().tobytes())]] += 1
        max_error = float(np.abs(counts / n_draws - exact.numpy()).max())
        self.assertLess(max_error, 0.006)

    def test4(self) -> None:
        """
        Relabelling the variables relabels utilities and logits.
        """
        torch.manual_seed(3)
        policy = tpolic.TukeyPolicy(hidden=16).double()
        obs, _ = _observation(6, seed=4)
        perm = [2, 5, 0, 1, 4, 3]
        u, logits = policy.decoder_params(obs)
        u_p, logits_p = policy.decoder_params(
            tfeat.permute_observation(obs, perm)
        )
        np.testing.assert_allclose(u_p.detach(), u.detach()[perm], atol=1e-10)
        np.testing.assert_allclose(
            logits_p.detach(), logits.detach()[perm][:, perm], atol=1e-10
        )

    def test5(self) -> None:
        """
        The topological order respects the edges and prefers larger utilities.
        """
        B = np.zeros((4, 4), dtype=int)
        B[2, 0] = 1
        B[2, 1] = 1
        u = np.array([0.0, 5.0, -1.0, 3.0])
        self.assertEqual(tpolic.topological_order(B, u), [3, 2, 1, 0])


class Test_tool_head(hunitest.TestCase):
    """
    Check the masked tool distribution.
    """

    def test1(self) -> None:
        """
        Illegal calls get probability 0; STOP stays legal; budget 0 forces STOP.
        """
        torch.manual_seed(4)
        policy = tpolic.TukeyPolicy(hidden=16)
        obs, evidence = _observation(5, seed=5, n_calls=7)
        actions, _ = policy.candidates(5)
        legal = tpolic.legal_mask(actions, evidence)
        log_probs = policy.tool_log_probs(obs, legal)
        self.assertEqual(int(legal.sum()), len(actions) - 7)
        self.assertTrue(
            torch.all(torch.isinf(log_probs[~torch.as_tensor(legal)]))
        )
        self.assertAlmostEqual(float(log_probs.exp().sum()), 1.0, places=5)
        self.assertEqual(actions[-1].kind, ttools.STOP)
        evidence.budget = evidence.calls_used
        legal = tpolic.legal_mask(actions, evidence)
        self.assertEqual(np.flatnonzero(legal).tolist(), [len(actions) - 1])

    def test2(self) -> None:
        """
        Relabelling the variables relabels the action probabilities.
        """
        torch.manual_seed(5)
        policy = tpolic.TukeyPolicy(hidden=16).double()
        obs, _ = _observation(5, seed=6, n_calls=0)
        actions, _ = policy.candidates(5)
        legal = np.ones(len(actions), dtype=bool)
        perm = [3, 0, 4, 1, 2]
        new_index = list(np.argsort(perm))
        log_probs = policy.tool_log_probs(obs, legal).detach()
        log_probs_p = policy.tool_log_probs(
            tfeat.permute_observation(obs, perm), legal
        ).detach()
        position = {a: k for k, a in enumerate(actions)}
        max_error = 0.0
        for k, action in enumerate(actions):
            args = [new_index[v] for v in action.args]
            if action.kind == ttools.CI_TEST:
                moved = ttools.ci_test_action(args[0], args[1], tuple(args[2:]))
            elif action.kind == ttools.REGRESS:
                moved = ttools.regress_action(args[0], tuple(args[1:]))
            elif action.kind == ttools.LINGAM_DIR:
                moved = ttools.Action(ttools.LINGAM_DIR, tuple(sorted(args)))
            else:
                moved = ttools.Action(action.kind, tuple(args))
            error = abs(float(log_probs[k] - log_probs_p[position[moved]]))
            max_error = max(max_error, error)
        self.assertLess(max_error, 1e-10)
