"""
Synthetic causal worlds with a known answer, and the Gym-style environment.

An instance is a hidden random DAG G*, data sampled from a structural equation
model on it, and a train/test split. The environment exposes D_train and D_test
to the experimenter (as the spec's Milestone 1 asks) but an agent only sees
`Evidence`: the free correlation matrix of D_train and the results of its own
tool calls. G* and the test rows only reach the scorer.

Families in this module:
- `proto`: the prototype's generator, bit-identical (exponential noise, raw
  scale, sequential split), to reproduce its numbers
- `linear_exp`, `linear_uniform`, `linear_gauss`: linear SEMs with one noise
  family; standardized with D_train statistics by default
- `linear_gauss_eqvar_raw`: Gaussian noise with equal variances on the raw
  scale, where the DAG is identifiable (Peters and Buhlmann 2014)
- `mixed`: one of the three noise families drawn per graph

Contracts of the linear families (checked in `test/test_tukey_env.py`):
- every node's noise explains at least 10% of its variance: incoming weights
  of a node are scaled down when needed (`cap_signal_share()`), since raw
  variance otherwise grows with depth
- every true edge is visible once the child's other parents are controlled for:
  population `|pcorr(i, j | Pa(j) - {i})| >= 0.1` (`passes_edge_margin()`);
  weights are redrawn until it holds
- other conditioning sets are not filtered: in dense graphs some set of size 2
  almost always makes a true edge look absent (Uhler et al. 2013), so the
  smallest such partial correlation is reported as a difficulty covariate
  (`min_partial_corr()`)

Import as:

import tukey_env as tenv
"""

import dataclasses
import itertools
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import helpers.hdbg as hdbg
import research.agentic_data_science.RL_for_auto_EDA_prototype as rlproto
import tukey_scoring as tscore
import tukey_tools as ttools

_LOG = logging.getLogger(__name__)

# Noise family of each single-family linear generator.
_FAMILY_NOISE = {
    "linear_exp": "exponential",
    "linear_uniform": "uniform",
    "linear_gauss": "gaussian",
    "linear_gauss_eqvar_raw": "gaussian",
}
_MIXED_NOISES = ("exponential", "uniform", "gaussian")
FAMILIES = ("proto", "mixed") + tuple(_FAMILY_NOISE)

# Noise-share floor and edge-strength margin of the linear families.
_MIN_NOISE_SHARE = 0.1
_MIN_EDGE_PCORR = 0.1
_MAX_TRIES = 500


# #############################################################################
# Configuration and instance
# #############################################################################


@dataclasses.dataclass(frozen=True)
class GraphConfig:
    """
    Settings of the instance generator.
    """

    # Number of variables.
    n_nodes: int = 6
    # Number of edges (capped at n_nodes * (n_nodes - 1) / 2).
    n_edges: int = 6
    # Generator family, one of `FAMILIES`.
    family: str = "mixed"
    # Noise scales: "random" draws U(0.5, 1.5) per node, "equal" uses 1.
    noise_scale: str = "random"
    # "post" standardizes with D_train statistics, "none" keeps the raw scale.
    standardize: str = "post"
    # Absolute edge weights are drawn uniformly from this range.
    w_range: Tuple[float, float] = (0.5, 2.0)
    # Number of rows, split into train and test.
    n_samples: int = 3000
    test_frac: float = 0.33


# The prototype's setting: 6 nodes, 6 edges, exponential noise, raw scale.
PROTO_CONFIG = GraphConfig(
    family="proto", noise_scale="equal", standardize="none"
)
# Gaussian noise with equal variances on the raw scale: the DAG is identifiable.
EQVAR_RAW_CONFIG = GraphConfig(
    family="linear_gauss_eqvar_raw", noise_scale="equal", standardize="none"
)


@dataclasses.dataclass(frozen=True)
class Instance:
    """
    One sampled world.

    Everything except `cfg`, `d_train` and `d_test` is private to the
    environment, the scorer and the reports.
    """

    cfg: GraphConfig
    seed: int
    d_train: pd.DataFrame
    d_test: pd.DataFrame
    truth: tscore.Truth
    # Weighted adjacency: `W[i, j]` is the effect of i on j.
    W: np.ndarray
    # Noise standard deviation and family of each node.
    noise_std: np.ndarray
    noise_families: Tuple[str, ...]
    # Why the target class was chosen, for the logs and the report.
    target_reason: str
    # Number of weight draws rejected by the edge-strength margin.
    n_rejected: int


# #############################################################################
# Generators
# #############################################################################


def _topological_order(B: np.ndarray) -> List[int]:
    """
    Return a topological order of a DAG.

    :param B: DAG adjacency
    :return: nodes with every parent before its children
    """
    in_degree = B.sum(axis=0).astype(int)
    order: List[int] = []
    ready = sorted(np.flatnonzero(in_degree == 0).tolist())
    while ready:
        node = ready.pop(0)
        order.append(node)
        for child in np.flatnonzero(B[node]):
            in_degree[child] -= 1
            if in_degree[child] == 0:
                ready.append(int(child))
    hdbg.dassert_eq(len(order), B.shape[0], "Graph has a cycle")
    return order


def _sample_er_dag(
    rng: np.random.Generator, n_nodes: int, n_edges: int
) -> np.ndarray:
    """
    Sample a DAG with exactly `n_edges` edges, as the prototype does.

    A random order is drawn and `n_edges` of the order-consistent pairs are
    chosen uniformly without replacement.

    :param rng: random generator
    :param n_nodes: number of nodes
    :param n_edges: number of edges
    :return: DAG adjacency
    """
    order = rng.permutation(n_nodes)
    pairs = [
        (order[i], order[j])
        for i, j in itertools.combinations(range(n_nodes), 2)
    ]
    n_edges = min(n_edges, len(pairs))
    B = np.zeros((n_nodes, n_nodes), dtype=int)
    for idx in rng.choice(len(pairs), size=n_edges, replace=False):
        i, j = pairs[idx]
        B[i, j] = 1
    return B


def population_covariance(W: np.ndarray, noise_var: np.ndarray) -> np.ndarray:
    """
    Compute the covariance of a linear SEM `x = W.T x + e`.

    :param W: weighted adjacency (`W[i, j]` is the effect of i on j)
    :param noise_var: noise variance of each node
    :return: covariance matrix of x
    """
    mixing = np.linalg.inv(np.eye(W.shape[0]) - W.T)
    cov = mixing @ np.diag(noise_var) @ mixing.T
    return cov


def _partial_corr(corr: np.ndarray, i: int, j: int, cond: List[int]) -> float:
    """
    Compute a partial correlation from a correlation matrix.

    :param corr: correlation matrix
    :param i: first variable
    :param j: second variable
    :param cond: conditioning set
    :return: partial correlation of i and j given `cond`
    """
    idx = [i, j] + cond
    precision = np.linalg.inv(corr[np.ix_(idx, idx)])
    pcorr = -precision[0, 1] / np.sqrt(precision[0, 0] * precision[1, 1])
    return float(pcorr)


def _population_corr(W: np.ndarray, noise_var: np.ndarray) -> np.ndarray:
    """
    Compute the correlation matrix of a linear SEM.

    :param W: weighted adjacency
    :param noise_var: noise variance of each node
    :return: correlation matrix of x
    """
    cov = population_covariance(W, noise_var)
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    return corr


def cap_signal_share(
    W: np.ndarray,
    noise_var: np.ndarray,
    *,
    min_noise_share: float = _MIN_NOISE_SHARE,
) -> np.ndarray:
    """
    Scale down the incoming weights of nodes whose noise share is too small.

    Nodes are processed in topological order. When a node's noise would
    explain less than `min_noise_share` of its variance, all its incoming
    weights are multiplied by the factor that brings the share to the floor,
    so every node's R2 on its parents is at most `1 - min_noise_share`.

    :param W: weighted adjacency
    :param noise_var: noise variance of each node
    :param min_noise_share: floor on `noise_var / Var(x)`
    :return: capped copy of `W`
    """
    W = W.copy()
    n_nodes = W.shape[0]
    cov = np.zeros((n_nodes, n_nodes))
    max_ratio = (1 - min_noise_share) / min_noise_share
    for j in _topological_order((W != 0).astype(int)):
        signal_var = W[:, j] @ cov @ W[:, j]
        max_signal_var = noise_var[j] * max_ratio
        if signal_var > max_signal_var:
            W[:, j] *= np.sqrt(max_signal_var / signal_var)
            signal_var = W[:, j] @ cov @ W[:, j]
        # Parents come first, so their covariances are final.
        cov[:, j] = cov @ W[:, j]
        cov[j, :] = cov[:, j]
        cov[j, j] = signal_var + noise_var[j]
    return W


def edge_strengths(W: np.ndarray, noise_var: np.ndarray) -> np.ndarray:
    """
    Compute each true edge's partial correlation given the child's other parents.

    :param W: weighted adjacency
    :param noise_var: noise variance of each node
    :return: `|pcorr(i, j | Pa(j) - {i})|` per edge, in `np.argwhere(W)` order
    """
    corr = _population_corr(W, noise_var)
    strengths = []
    for i, j in np.argwhere(W != 0):
        others = [int(k) for k in np.flatnonzero(W[:, j]) if k != i]
        strengths.append(abs(_partial_corr(corr, i, j, others)))
    return np.array(strengths)


def passes_edge_margin(W: np.ndarray, noise_var: np.ndarray) -> bool:
    """
    Check that every true edge is visible given the child's other parents.

    :param W: weighted adjacency
    :param noise_var: noise variance of each node
    :return: True if every edge strength is at least 0.1
    """
    passes = bool(np.all(edge_strengths(W, noise_var) >= _MIN_EDGE_PCORR))
    return passes


def min_partial_corr(
    W: np.ndarray, noise_var: np.ndarray, *, max_cond: int = 2
) -> float:
    """
    Find the smallest partial correlation of a true edge over small sets.

    A value near 0 means some test with at most `max_cond` conditioning
    variables makes a true edge look absent (a near-unfaithful set).

    :param W: weighted adjacency
    :param noise_var: noise variance of each node
    :param max_cond: largest conditioning set
    :return: `min |pcorr(i, j | S)|` over edges i -> j and `|S| <= max_cond`
    """
    corr = _population_corr(W, noise_var)
    n_nodes = W.shape[0]
    smallest = 1.0
    for i, j in np.argwhere(W != 0):
        others = [k for k in range(n_nodes) if k not in (i, j)]
        for size in range(max_cond + 1):
            for cond in itertools.combinations(others, size):
                pcorr = _partial_corr(corr, i, j, list(cond))
                smallest = min(smallest, abs(pcorr))
    return smallest


def _sample_noise(
    rng: np.random.Generator, family: str, n_samples: int, scale: float
) -> np.ndarray:
    """
    Sample centred noise with standard deviation `scale`.

    :param rng: random generator
    :param family: "exponential", "uniform" or "gaussian"
    :param n_samples: number of draws
    :param scale: standard deviation
    :return: noise vector
    """
    if family == "exponential":
        noise = rng.exponential(1.0, size=n_samples) - 1.0
    elif family == "uniform":
        noise = rng.uniform(-np.sqrt(3), np.sqrt(3), size=n_samples)
    elif family == "gaussian":
        noise = rng.normal(size=n_samples)
    else:
        raise ValueError(f"Unknown noise family '{family}'")
    return scale * noise


def _simulate_linear(
    rng: np.random.Generator,
    W: np.ndarray,
    families: Tuple[str, ...],
    noise_std: np.ndarray,
    n_samples: int,
) -> np.ndarray:
    """
    Sample rows of a linear SEM in topological order.

    :param rng: random generator
    :param W: weighted adjacency
    :param families: noise family per node
    :param noise_std: noise standard deviation per node
    :param n_samples: number of rows
    :return: data, shape (n_samples, n_nodes)
    """
    n_nodes = W.shape[0]
    x = np.zeros((n_samples, n_nodes))
    for j in _topological_order((W != 0).astype(int)):
        noise = _sample_noise(rng, families[j], n_samples, noise_std[j])
        x[:, j] = x @ W[:, j] + noise
    return x


def _target_for_linear(
    families: Tuple[str, ...], cfg: GraphConfig
) -> Tuple[str, str]:
    """
    Decide what is identifiable for a linear SEM instance.

    :param families: noise family per node
    :param cfg: generator settings
    :return: target class and the reason
    """
    gaussian = [f == "gaussian" for f in families]
    if not any(gaussian):
        target = (tscore.TARGET_DAG, "linear, every noise non-Gaussian (LiNGAM)")
    elif all(gaussian):
        equal_raw = cfg.noise_scale == "equal" and cfg.standardize == "none"
        if equal_raw:
            target = (
                tscore.TARGET_DAG,
                "linear Gaussian, equal noise variances, raw scale",
            )
        else:
            target = (
                tscore.TARGET_CPDAG,
                "linear Gaussian: equivalence class only",
            )
    else:
        target = (tscore.TARGET_NGDAG, "mixed Gaussian and non-Gaussian noise")
    return target


def _split_and_scale(
    x: np.ndarray,
    cfg: GraphConfig,
    rng_split: Optional[np.random.Generator],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split rows into train and test, then standardize with train statistics.

    :param x: data
    :param cfg: generator settings
    :param rng_split: random generator of the split; None keeps the row order
        (the prototype's sequential split)
    :return: training rows and test rows
    """
    n_samples = x.shape[0]
    cut = int(n_samples * (1 - cfg.test_frac))
    if rng_split is None:
        rows = np.arange(n_samples)
    else:
        rows = rng_split.permutation(n_samples)
    x_train, x_test = x[rows[:cut]], x[rows[cut:]]
    if cfg.standardize == "post":
        mean, std = x_train.mean(axis=0), x_train.std(axis=0)
        x_train = (x_train - mean) / std
        x_test = (x_test - mean) / std
    else:
        hdbg.dassert_eq(cfg.standardize, "none")
    return x_train, x_test


def _sample_proto(cfg: GraphConfig, seed: int) -> Dict[str, Any]:
    """
    Sample an instance exactly as the prototype's `Environment` does.

    :param cfg: generator settings (family "proto", raw scale)
    :param seed: the prototype's seed
    :return: fields of `Instance` besides `cfg`, `seed`, data and `truth`
    """
    hdbg.dassert_eq(cfg.standardize, "none", "The prototype's data is raw")
    W = rlproto.random_dag(cfg.n_nodes, cfg.n_edges, seed, w_range=cfg.w_range)
    x = rlproto.simulate_lingam(W, cfg.n_samples, seed)
    x_train, x_test = _split_and_scale(x, cfg, None)
    sample = {
        "W": W,
        "x_train": x_train,
        "x_test": x_test,
        # Centred Exp(1) noise has standard deviation 1.
        "noise_std": np.ones(cfg.n_nodes),
        "noise_families": ("exponential",) * cfg.n_nodes,
        "target_class": tscore.TARGET_DAG,
        "target_reason": "prototype (LiNGAM)",
        "n_rejected": 0,
    }
    return sample


def _sample_linear(cfg: GraphConfig, seed: int) -> Dict[str, Any]:
    """
    Sample a linear-SEM instance that meets the family contracts.

    :param cfg: generator settings
    :param seed: instance seed
    :return: fields of `Instance` besides `cfg`, `seed`, data and `truth`
    """
    if cfg.family == "linear_gauss_eqvar_raw":
        hdbg.dassert_eq(cfg.noise_scale, "equal", "Use EQVAR_RAW_CONFIG")
        hdbg.dassert_eq(cfg.standardize, "none", "Use EQVAR_RAW_CONFIG")
    # The same seed gives the same graph and weights in every linear family,
    # so families can be compared on paired instances.
    rng_graph, rng_weights, rng_noise, rng_split, rng_perm = [
        np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(5)
    ]
    B = _sample_er_dag(rng_graph, cfg.n_nodes, cfg.n_edges)
    if cfg.family == "mixed":
        family = _MIXED_NOISES[rng_graph.integers(len(_MIXED_NOISES))]
    else:
        family = _FAMILY_NOISE[cfg.family]
    families = (family,) * cfg.n_nodes
    # Draw weights and noise scales, cap the signal shares and redraw until
    # every edge passes the margin.
    n_rejected = 0
    for _ in range(_MAX_TRIES):
        signs = rng_weights.choice([-1.0, 1.0], size=B.shape)
        W = B * signs * rng_weights.uniform(*cfg.w_range, size=B.shape)
        if cfg.noise_scale == "equal":
            noise_std = np.ones(cfg.n_nodes)
        else:
            noise_std = rng_weights.uniform(0.5, 1.5, size=cfg.n_nodes)
        W = cap_signal_share(W, noise_std**2)
        if passes_edge_margin(W, noise_std**2):
            break
        n_rejected += 1
    else:
        raise RuntimeError(f"No weights passed the margin in {_MAX_TRIES} tries")
    x = _simulate_linear(rng_noise, W, families, noise_std, cfg.n_samples)
    # Permute the columns so that column order cannot leak the causal order.
    perm = rng_perm.permutation(cfg.n_nodes)
    x = x[:, perm]
    x_train, x_test = _split_and_scale(x, cfg, rng_split)
    target_class, target_reason = _target_for_linear(families, cfg)
    sample = {
        "W": W[np.ix_(perm, perm)],
        "x_train": x_train,
        "x_test": x_test,
        "noise_std": noise_std[perm],
        "noise_families": tuple(families[k] for k in perm),
        "target_class": target_class,
        "target_reason": target_reason,
        "n_rejected": n_rejected,
    }
    return sample


def sample_instance(cfg: GraphConfig, seed: int) -> Instance:
    """
    Sample a hidden graph, its data and the train/test split.

    :param cfg: generator settings
    :param seed: instance seed (same seed and settings give the same instance)
    :return: the instance
    """
    hdbg.dassert_in(cfg.family, FAMILIES)
    if cfg.family == "proto":
        sample = _sample_proto(cfg, seed)
    else:
        sample = _sample_linear(cfg, seed)
    B = (sample["W"] != 0).astype(int)
    nongauss = np.array([f != "gaussian" for f in sample["noise_families"]])
    truth = tscore.make_truth(
        B, sample["target_class"], nongauss, sample["x_train"], sample["x_test"]
    )
    columns = [f"x{k}" for k in range(cfg.n_nodes)]
    instance = Instance(
        cfg=cfg,
        seed=seed,
        d_train=pd.DataFrame(sample["x_train"], columns=columns),
        d_test=pd.DataFrame(sample["x_test"], columns=columns),
        truth=truth,
        W=sample["W"],
        noise_std=sample["noise_std"],
        noise_families=sample["noise_families"],
        target_reason=sample["target_reason"],
        n_rejected=sample["n_rejected"],
    )
    return instance


# #############################################################################
# TukeyEnv
# #############################################################################


class TukeyEnv:
    """
    Gym-style environment: hidden graph, tool calls on D_train, scored emit.

    `reset()` returns `(evidence, info)`; `step(action)` runs one tool call and
    returns `(evidence, reward, terminated, truncated, info)`, where the reward
    is the call's training cost (negative). `submit_graph(edges)` ends the
    episode and returns the score. Signatures follow Gymnasium; the
    `gymnasium` package itself is not used because observations have a
    variable size.
    """

    def __init__(
        self, cfg: GraphConfig, *, budget: Optional[int] = None
    ) -> None:
        """
        Create an environment for one generator configuration.

        :param cfg: generator settings
        :param budget: tool-call budget per episode (0 allows no tool call)
            - Default: None means `3 * n_nodes`
        """
        self.cfg = cfg
        self.budget = 3 * cfg.n_nodes if budget is None else budget
        hdbg.dassert_lte(0, self.budget)
        self._instance: Optional[Instance] = None
        self._evidence: Optional[ttools.Evidence] = None
        self._all_actions = ttools.enumerate_tool_actions(cfg.n_nodes)
        self._done = True

    @property
    def d_train(self) -> pd.DataFrame:
        """
        Training rows of the current instance (the experimenter's view).
        """
        hdbg.dassert_is_not(self._instance, None, "Call reset() first")
        return self._instance.d_train

    @property
    def d_test(self) -> pd.DataFrame:
        """
        Test rows of the current instance (the scorer's data, never an agent's).
        """
        hdbg.dassert_is_not(self._instance, None, "Call reset() first")
        return self._instance.d_test

    def reset(self, *, seed: int) -> Tuple[ttools.Evidence, Dict[str, Any]]:
        """
        Sample a new instance and start an episode.

        :param seed: instance seed
        :return: the agent's evidence and `info` (sizes and costs only)
        """
        self._instance = sample_instance(self.cfg, seed)
        cache = ttools.ToolCache(self._instance.d_train.to_numpy())
        self._evidence = ttools.Evidence(cache, self.budget)
        self._done = False
        info = {
            "n_nodes": self.cfg.n_nodes,
            "n_train": cache.n_train,
            "budget": self.budget,
            "tool_costs": dict(ttools.TOOL_COSTS),
        }
        return self._evidence, info

    def legal_actions(self) -> List[ttools.Action]:
        """
        List the tool calls still allowed, plus STOP.

        :return: legal actions
        """
        hdbg.dassert(not self._done, "Episode is over: call reset()")
        legal = self._evidence.legal_tool_actions(self._all_actions)
        legal.append(ttools.Action(ttools.STOP))
        return legal

    def step(
        self, action: ttools.Action
    ) -> Tuple[ttools.Evidence, float, bool, bool, Dict[str, Any]]:
        """
        Run one tool call, or STOP.

        :param action: a legal action
        :return: evidence, reward (training cost of the call), terminated
            (True after STOP), truncated (True once the budget is spent), info
        """
        hdbg.dassert(not self._done, "Episode is over: call reset()")
        if action.kind == ttools.STOP:
            self._done = True
            return self._evidence, 0.0, True, False, {}
        self._evidence.record(action)
        reward = -ttools.TOOL_COSTS[action.kind]
        truncated = self._evidence.budget_left == 0
        return self._evidence, reward, False, truncated, {}

    def submit_graph(self, edges: List[Tuple[int, int]]) -> Dict[str, Any]:
        """
        Score a discovered graph (the prototype's `graph.json` contract).

        :param edges: `[(parent, child), ...]`
        :return: score dictionary of `tukey_scoring.score_graph()`, plus the
            number of tool calls used and of dropped edges
        """
        hdbg.dassert_is_not(self._instance, None, "Call reset() first")
        B_hat, n_dropped = tscore.edges_to_adjacency(edges, self.cfg.n_nodes)
        scores = tscore.score_graph(B_hat, self._instance.truth)
        scores["n_dropped_edges"] = n_dropped
        scores["calls_used"] = self._evidence.calls_used
        self._done = True
        return scores
