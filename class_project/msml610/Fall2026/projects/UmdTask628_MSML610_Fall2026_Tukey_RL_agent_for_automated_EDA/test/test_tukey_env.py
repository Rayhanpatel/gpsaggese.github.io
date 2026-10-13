"""
Test the Tukey environment: generator contracts, prototype parity, leaks.

This is integration gate G3 and the environment part of G8. The `proto` family
must reproduce the prototype bit for bit; the linear families must meet their
contracts (acyclic, exact edge counts, noise-share floor, edge-strength margin,
train-only standardization, per-instance target class); nothing an agent
receives may carry the hidden graph or the test rows.

Import as:

import test.test_tukey_env as ttetuenv
"""

import collections
import dataclasses
import logging
from typing import Dict, List

import numpy as np
import scipy.stats

import helpers.hunit_test as hunitest
import research.agentic_data_science.RL_for_auto_EDA_prototype as rlproto
import tukey_env as tenv
import tukey_scoring as tscore
import tukey_tools as ttools

_LOG = logging.getLogger(__name__)

_LINEAR_CONFIGS: Dict[str, tenv.GraphConfig] = {
    "linear_exp": tenv.GraphConfig(family="linear_exp"),
    "linear_uniform": tenv.GraphConfig(family="linear_uniform"),
    "linear_gauss": tenv.GraphConfig(family="linear_gauss"),
    "linear_gauss_eqvar_raw": tenv.EQVAR_RAW_CONFIG,
    "mixed": tenv.GraphConfig(family="mixed"),
}


# #############################################################################
# Fixtures
# #############################################################################


def _check_contracts(test: hunitest.TestCase, inst: tenv.Instance) -> None:
    """
    Assert the contracts every linear-family instance must meet.

    :param test: calling test case
    :param inst: sampled instance
    """
    cfg = inst.cfg
    B = inst.truth.B
    x_train = inst.d_train.to_numpy()
    x_test = inst.d_test.to_numpy()
    test.assertTrue(tscore.is_dag(B))
    test.assertEqual(int(B.sum()), cfg.n_edges)
    test.assertTrue(np.array_equal(B, (inst.W != 0).astype(int)))
    # `int(3000 * (1 - 0.33))` is 2009 in floating point, as in the prototype.
    test.assertEqual(x_train.shape, (2009, cfg.n_nodes))
    test.assertEqual(x_test.shape, (991, cfg.n_nodes))
    test.assertTrue(np.isfinite(x_train).all() and np.isfinite(x_test).all())
    if cfg.standardize == "post":
        test.assertLess(np.abs(x_train.mean(axis=0)).max(), 1e-12)
        test.assertLess(np.abs(x_train.std(axis=0) - 1).max(), 1e-12)
    noise_var = inst.noise_std**2
    cov = tenv.population_covariance(inst.W, noise_var)
    test.assertGreaterEqual((noise_var / np.diag(cov)).min(), 0.1 - 1e-9)
    test.assertGreaterEqual(tenv.edge_strengths(inst.W, noise_var).min(), 0.1)
    nongauss = np.array([f != "gaussian" for f in inst.noise_families])
    test.assertTrue(np.array_equal(inst.truth.nongauss, nongauss))


def _chain(weights: List[float]) -> np.ndarray:
    """
    Build the weighted adjacency of a chain 0 -> 1 -> ... with given weights.

    :param weights: weight of each consecutive edge
    :return: weighted adjacency
    """
    n_nodes = len(weights) + 1
    W = np.zeros((n_nodes, n_nodes))
    for k, weight in enumerate(weights):
        W[k, k + 1] = weight
    return W


# #############################################################################
# Test_sample_instance_proto
# #############################################################################


class Test_sample_instance_proto(hunitest.TestCase):
    """
    The `proto` family reproduces the prototype exactly.
    """

    def test1(self) -> None:
        """
        Seeds 0-19 give the prototype's weights, graph and split bit for bit.
        """
        for seed in range(20):
            inst = tenv.sample_instance(tenv.PROTO_CONFIG, seed)
            env = rlproto.Environment(seed=seed)
            self.assertTrue(np.array_equal(inst.W, env.W))
            self.assertTrue(np.array_equal(inst.truth.B, env.B_true))
            self.assertTrue(np.array_equal(inst.d_train.to_numpy(), env.X_train))
            self.assertTrue(np.array_equal(inst.d_test.to_numpy(), env.X_test))

    def test2(self) -> None:
        """
        The scorer's legacy keys equal the prototype's `score()` dictionaries.
        """
        for seed in range(5):
            inst = tenv.sample_instance(tenv.PROTO_CONFIG, seed)
            env = rlproto.Environment(seed=seed)
            for B_hat in (env.B_true, np.zeros_like(env.B_true), env.B_true.T):
                scores = tscore.score_graph(B_hat, inst.truth)
                expected = rlproto.score(B_hat, env)
                actual = {key: scores[key] for key in expected}
                self.assertEqual(actual, expected)

    def test3(self) -> None:
        """
        The prototype family refuses standardization.
        """
        cfg = dataclasses.replace(tenv.PROTO_CONFIG, standardize="post")
        with self.assertRaises(AssertionError):
            tenv.sample_instance(cfg, 0)


# #############################################################################
# Test_sample_instance_linear
# #############################################################################


class Test_sample_instance_linear(hunitest.TestCase):
    """
    The linear families meet their contracts.
    """

    def test1(self) -> None:
        """
        Every family meets the contracts on 50 seeds; targets follow the noise.
        """
        lines = []
        for name, cfg in _LINEAR_CONFIGS.items():
            targets: collections.Counter = collections.Counter()
            n_rejected = 0
            for seed in range(50):
                inst = tenv.sample_instance(cfg, seed)
                _check_contracts(self, inst)
                targets[inst.truth.target_class] += 1
                n_rejected += inst.n_rejected
            lines.append(f"{name}: {dict(sorted(targets.items()))} {n_rejected}")
        actual = "\n".join(lines)
        expected = """
        linear_exp: {'dag': 50} 0
        linear_uniform: {'dag': 50} 0
        linear_gauss: {'cpdag': 50} 0
        linear_gauss_eqvar_raw: {'dag': 50} 0
        mixed: {'cpdag': 15, 'dag': 35} 0
        """
        self.assert_equal(actual, expected, dedent=True)

    def test2(self) -> None:
        """
        The same seed gives the same instance; another seed a different one.
        """
        cfg = _LINEAR_CONFIGS["mixed"]
        inst1 = tenv.sample_instance(cfg, 7)
        inst2 = tenv.sample_instance(cfg, 7)
        inst3 = tenv.sample_instance(cfg, 8)
        self.assertTrue(inst1.d_train.equals(inst2.d_train))
        self.assertTrue(np.array_equal(inst1.W, inst2.W))
        self.assertFalse(inst1.d_train.equals(inst3.d_train))

    def test3(self) -> None:
        """
        Families share graph and weights per seed and differ only in noise.
        """
        for seed in range(10):
            inst_exp = tenv.sample_instance(_LINEAR_CONFIGS["linear_exp"], seed)
            inst_gauss = tenv.sample_instance(
                _LINEAR_CONFIGS["linear_gauss"], seed
            )
            self.assertTrue(np.array_equal(inst_exp.W, inst_gauss.W))
            self.assertFalse(inst_exp.d_train.equals(inst_gauss.d_train))

    def test4(self) -> None:
        """
        The sampled noise has the family's shape (raw scale, true weights).
        """
        lines = []
        for family in ("linear_exp", "linear_uniform", "linear_gauss"):
            cfg = tenv.GraphConfig(
                family=family, noise_scale="equal", standardize="none"
            )
            inst = tenv.sample_instance(cfg, 11)
            x = inst.d_train.to_numpy()
            noise = x - x @ inst.W
            skew = scipy.stats.skew(noise, axis=0)
            kurt = scipy.stats.kurtosis(noise, axis=0)
            lines.append(
                f"{family}: skew {skew.min():.1f}..{skew.max():.1f}, "
                f"kurt {kurt.min():.1f}..{kurt.max():.1f}"
            )
        actual = "\n".join(lines)
        expected = """
        linear_exp: skew 1.6..1.9, kurt 3.3..5.5
        linear_uniform: skew -0.0..0.0, kurt -1.2..-1.2
        linear_gauss: skew -0.1..0.0, kurt -0.1..0.3
        """
        self.assert_equal(actual, expected, dedent=True)

    def test5(self) -> None:
        """
        Column order does not reveal edge direction (500 graphs).
        """
        n_forward = 0
        n_edges = 0
        for seed in range(500):
            B = tenv.sample_instance(_LINEAR_CONFIGS["linear_exp"], seed).truth.B
            parents, children = np.nonzero(B)
            n_forward += int((parents < children).sum())
            n_edges += len(parents)
        frac = n_forward / n_edges
        _LOG.info("Edges from a lower to a higher column: %.3f", frac)
        self.assertLess(abs(frac - 0.5), 0.05)

    def test6(self) -> None:
        """
        The equal-variance raw family refuses other settings.
        """
        cfg = tenv.GraphConfig(family="linear_gauss_eqvar_raw")
        with self.assertRaises(AssertionError):
            tenv.sample_instance(cfg, 0)


# #############################################################################
# Test_generator_checks
# #############################################################################


class Test_generator_checks(hunitest.TestCase):
    """
    Check the signal cap, the edge margin and the difficulty covariate.
    """

    def test1(self) -> None:
        """
        The cap scales down deep weights of a strong chain to a 10% noise share.
        """
        W = _chain([2.0, 2.0, 2.0])
        noise_var = np.ones(4)
        W_capped = tenv.cap_signal_share(W, noise_var)
        cov = tenv.population_covariance(W_capped, noise_var)
        shares = noise_var / np.diag(cov)
        actual = (
            f"weights {np.round(np.diag(W_capped, k=1), 4).tolist()}\n"
            f"shares {np.round(shares, 4).tolist()}"
        )
        expected = """
        weights [2.0, 1.3416, 0.9487]
        shares [1.0, 0.2, 0.1, 0.1]
        """
        self.assert_equal(actual, expected, dedent=True)

    def test2(self) -> None:
        """
        The cap leaves a weak chain unchanged.
        """
        W = _chain([0.8, 0.8, 0.8])
        W_capped = tenv.cap_signal_share(W, np.ones(4))
        self.assertTrue(np.array_equal(W, W_capped))

    def test3(self) -> None:
        """
        Path cancellation: edges pass the margin, the marginal correlation is 0.

        In 0 -> 1 -> 2 with 0 -> 2 of weight -1, x2 = e1 + e2, so x0 and x2 are
        uncorrelated although 0 -> 2 is an edge; given x1 it is visible.
        """
        W = np.zeros((3, 3))
        W[0, 1] = 1.0
        W[1, 2] = 1.0
        W[0, 2] = -1.0
        noise_var = np.ones(3)
        strengths = tenv.edge_strengths(W, noise_var)
        actual = (
            f"strengths {np.round(strengths, 4).tolist()}\n"
            f"margin {tenv.passes_edge_margin(W, noise_var)}\n"
            f"min marginal {tenv.min_partial_corr(W, noise_var, max_cond=0):.4f}"
        )
        expected = """
        strengths [0.7071, 0.5774, 0.7071]
        margin True
        min marginal 0.0000
        """
        self.assert_equal(actual, expected, dedent=True)

    def test4(self) -> None:
        """
        A weak edge into a noisy child fails the margin.
        """
        W = np.zeros((2, 2))
        W[0, 1] = 0.05
        self.assertFalse(tenv.passes_edge_margin(W, np.ones(2)))


# #############################################################################
# Test_TukeyEnv
# #############################################################################


class Test_TukeyEnv(hunitest.TestCase):
    """
    Check the episode API and what reaches an agent.
    """

    def test1(self) -> None:
        """
        Reset gives evidence built from D_train only and a whitelisted `info`.
        """
        env = tenv.TukeyEnv(_LINEAR_CONFIGS["linear_exp"])
        evidence, info = env.reset(seed=3)
        self.assertEqual(
            sorted(info), ["budget", "n_nodes", "n_train", "tool_costs"]
        )
        self.assertEqual(info["budget"], 18)
        x_train = env.d_train.to_numpy()
        self.assertTrue(
            np.array_equal(evidence.corr, np.corrcoef(x_train, rowvar=False))
        )
        self.assertEqual(
            sorted(vars(evidence)),
            [
                "_cache",
                "budget",
                "corr",
                "history",
                "n_nodes",
                "n_train",
                "results",
            ],
        )
        # The only data the tools hold are the training rows.
        arrays = [
            v for v in vars(evidence._cache).values() if hasattr(v, "shape")
        ]
        for array in arrays:
            self.assertNotEqual(array.shape, env.d_test.shape)
        self.assertTrue(np.array_equal(evidence._cache._x, x_train))

    def test2(self) -> None:
        """
        Every tool result is reproduced by a fresh cache on D_train alone.
        """
        env = tenv.TukeyEnv(_LINEAR_CONFIGS["mixed"], budget=300)
        evidence, _ = env.reset(seed=4)
        fresh = ttools.ToolCache(env.d_train.to_numpy())
        for action in ttools.enumerate_tool_actions(6):
            self.assertEqual(evidence.record(action), fresh.run(action))

    def test3(self) -> None:
        """
        Steps cost their price; budget, repeats, STOP and reuse are enforced.
        """
        env = tenv.TukeyEnv(_LINEAR_CONFIGS["linear_gauss"], budget=2)
        env.reset(seed=5)
        action = ttools.ci_test_action(0, 1, (2,))
        _, reward, terminated, truncated, _ = env.step(action)
        self.assertEqual((reward, terminated, truncated), (-0.001, False, False))
        with self.assertRaises(AssertionError):
            env.step(action)
        _, _, _, truncated, _ = env.step(ttools.Action(ttools.NONGAUSS, (0,)))
        self.assertTrue(truncated)
        self.assertEqual(env.legal_actions(), [ttools.Action(ttools.STOP)])
        _, reward, terminated, _, _ = env.step(ttools.Action(ttools.STOP))
        self.assertEqual((reward, terminated), (0.0, True))
        with self.assertRaises(AssertionError):
            env.step(ttools.Action(ttools.NONGAUSS, (1,)))

    def test4(self) -> None:
        """
        Submitting G* scores 0, the empty graph -1; bad edges are dropped.
        """
        env = tenv.TukeyEnv(_LINEAR_CONFIGS["linear_exp"])
        env.reset(seed=6)
        true_edges = [tuple(e) for e in np.argwhere(env._instance.truth.B == 1)]
        scores = env.submit_graph(true_edges + [(0, 0), (0, 9)])
        actual = (
            f"true: score {scores['score']}, shd_target {scores['shd_target']}, "
            f"dropped {scores['n_dropped_edges']}, calls {scores['calls_used']}"
        )
        env.reset(seed=6)
        scores = env.submit_graph([])
        actual += f"\nempty: score {scores['score']}, shd {scores['shd_target']}"
        expected = """
        true: score 0.0, shd_target 0, dropped 2, calls 0
        empty: score -1.0, shd 6
        """
        self.assert_equal(actual, expected, dedent=True)


# #############################################################################
# Test_more_families
# #############################################################################


_MORE_CONFIGS: Dict[str, tenv.GraphConfig] = {
    "mixed_node": tenv.GraphConfig(family="mixed_node"),
    "nonlinear": tenv.GraphConfig(family="nonlinear", n_nodes=8, n_edges=10),
    "iscm": tenv.GraphConfig(family="iscm"),
    "pgmpy_lg": tenv.GraphConfig(family="pgmpy_lg"),
    "sklearn_reg": tenv.GraphConfig(family="sklearn_reg", n_nodes=7, n_edges=3),
    "sklearn_confounder": tenv.GraphConfig(
        family="sklearn_confounder", n_nodes=6, n_edges=2, confounder="observed"
    ),
    "scale_free": tenv.GraphConfig(n_nodes=8, n_edges=12, topology="scale_free"),
    "chain": tenv.GraphConfig(n_nodes=8, n_edges=7, topology="chain"),
    "star": tenv.GraphConfig(n_nodes=8, n_edges=7, topology="star"),
    "layered": tenv.GraphConfig(n_nodes=9, n_edges=12, topology="layered"),
}


class Test_more_families(hunitest.TestCase):
    """
    Contracts and family-specific checks of the M1 families and topologies.
    """

    def test1(self) -> None:
        """
        Every family: acyclic, exact edge count, standardized, finite; targets.
        """
        lines = []
        for name, cfg in _MORE_CONFIGS.items():
            targets: collections.Counter = collections.Counter()
            for seed in range(20):
                inst = tenv.sample_instance(cfg, seed)
                B = inst.truth.B
                x_train = inst.d_train.to_numpy()
                self.assertTrue(tscore.is_dag(B))
                self.assertEqual(int(B.sum()), cfg.n_edges)
                self.assertTrue(np.isfinite(x_train).all())
                self.assertLess(np.abs(x_train.std(axis=0) - 1).max(), 1e-12)
                if inst.mechanism == "linear":
                    noise_var = inst.noise_std**2
                    cov = tenv.population_covariance(inst.W, noise_var)
                    share = (noise_var / np.diag(cov)).min()
                    self.assertGreaterEqual(share, 0.1 - 1e-9)
                targets[inst.truth.target_class] += 1
            lines.append(f"{name}: {dict(sorted(targets.items()))}")
        actual = "\n".join(lines)
        expected = """
        mixed_node: {'dag': 3, 'ngdag_pattern': 17}
        nonlinear: {'dag': 20}
        iscm: {'cpdag': 5, 'dag': 15}
        pgmpy_lg: {'cpdag': 20}
        sklearn_reg: {'cpdag': 20}
        sklearn_confounder: {'cpdag': 20}
        scale_free: {'cpdag': 4, 'dag': 16}
        chain: {'cpdag': 5, 'dag': 15}
        star: {'cpdag': 5, 'dag': 15}
        layered: {'cpdag': 6, 'dag': 14}
        """
        self.assert_equal(actual, expected, dedent=True)

    def test2(self) -> None:
        """
        Quadratic mechanism: a symmetric parent is uncorrelated with its child.
        """
        cfg = tenv.GraphConfig(
            family="nonlinear", n_nodes=2, n_edges=1, standardize="none"
        )
        found = 0
        for seed in range(200):
            inst = tenv.sample_instance(cfg, seed)
            symmetric = inst.noise_families[0] in ("gaussian", "uniform")
            if inst.mechanism != "quadratic" or not symmetric:
                continue
            (parent,), (child,) = np.nonzero(inst.truth.B)
            x = inst.d_train.to_numpy()
            linear = abs(np.corrcoef(x[:, parent], x[:, child])[0, 1])
            squared = abs(np.corrcoef(x[:, parent] ** 2, x[:, child])[0, 1])
            self.assertLess(linear, 0.1)
            self.assertGreater(squared, 0.2)
            found += 1
        self.assertGreater(found, 10)

    def test3(self) -> None:
        """
        iSCM: every variable has unit variance before any post-processing.
        """
        cfg = tenv.GraphConfig(
            family="iscm", n_nodes=8, n_edges=12, standardize="none"
        )
        for seed in range(10):
            x = tenv.sample_instance(cfg, seed).d_train.to_numpy()
            self.assertLess(np.abs(x.var(axis=0) - 1).max(), 0.15)

    def test4(self) -> None:
        """
        Observed common cause: T1 and T2 are dependent, independent given U.
        """
        cfg = _MORE_CONFIGS["sklearn_confounder"]
        for seed in range(10):
            inst = tenv.sample_instance(cfg, seed)
            (source,) = np.flatnonzero(inst.truth.B.sum(axis=1))
            t_1, t_2 = np.flatnonzero(inst.truth.B[source])
            cache = ttools.ToolCache(inst.d_train.to_numpy())
            marginal = cache.fisher_z(cache.corr[t_1, t_2], 0)[1]
            given_u = cache.run(ttools.ci_test_action(t_1, t_2, (source,)))
            self.assertLess(marginal, 1e-6)
            self.assertGreater(given_u["p"], 0.001)

    def test5(self) -> None:
        """
        Topologies: chain is a path, star has a hub, layers are short.
        """
        for seed in range(10):
            B = tenv.sample_instance(_MORE_CONFIGS["chain"], seed).truth.B
            self.assertEqual(sorted(B.sum(axis=0).tolist()), [0] + [1] * 7)
            self.assertEqual(sorted(B.sum(axis=1).tolist()), [0] + [1] * 7)
            B = tenv.sample_instance(_MORE_CONFIGS["star"], seed).truth.B
            self.assertEqual(int(B.sum(axis=1).max()), 7)
            B = tenv.sample_instance(_MORE_CONFIGS["layered"], seed).truth.B
            # 9 nodes in 3 layers: no directed path longer than 2 edges.
            self.assertEqual(int(np.linalg.matrix_power(B, 3).sum()), 0)
