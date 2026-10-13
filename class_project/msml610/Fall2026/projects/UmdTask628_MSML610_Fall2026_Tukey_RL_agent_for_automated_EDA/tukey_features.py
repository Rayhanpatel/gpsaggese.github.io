"""
Turn an agent's evidence into fixed-width node, pair and global features.

Features are built only from what the agent observed: the free correlation
matrix of D_train, the results of its own tool calls, and the budget. Every
tool group is encoded as `(value * mask, mask)`, so "not measured" is distinct
from "measured as zero". Counts are fractions, so the width does not depend on
the number of variables. Partial correlations are never precomputed from the
free matrix: they enter only through `ci_test` calls.

Pair features are for ordered pairs `(i, j)` read as "i -> j"; symmetric
statistics are copied to both orders and directional ones are antisymmetric.

Import as:

import tukey_features as tfeat
"""

import dataclasses
import hashlib
import logging
import math
from typing import List

import numpy as np

import tukey_tools as ttools

_LOG = logging.getLogger(__name__)

NODE_FEATURES = (
    "nongauss_mask",
    "skew",
    "kurt",
    "jb_neglog_p",
    "variance_mask",
    "log_var",
    "regress_target_mask",
    "best_r2_as_target",
    "frac_pairs_ci_tested",
    "frac_pairs_lingam_tested",
    "mean_abs_r",
    "max_abs_r",
)
PAIR_FEATURES = (
    "abs_r",
    "r",
    "fisher_z",
    "triad_abs_r",
    "ci_mask",
    "ci_cover_1",
    "ci_cover_2",
    "min_abs_pcorr",
    "indep_p01",
    "indep_p05",
    "best_sep_size",
    "regress_fwd_mask",
    "regress_fwd_t",
    "regress_fwd_dep",
    "regress_fwd_ad",
    "regress_bwd_mask",
    "regress_bwd_dep",
    "proto_dir",
    "lingam_mask",
    "lingam_dir",
)
GLOBAL_FEATURES = (
    "log_n_nodes",
    "log_n_train",
    "budget_left_frac",
    "calls_used_frac",
    "log_budget",
    "frac_pairs_ci_tested",
    "frac_nodes_nongauss_tested",
    "mean_abs_skew",
    "mean_abs_kurt",
    "mean_abs_r",
)
_NODE = {name: k for k, name in enumerate(NODE_FEATURES)}
_PAIR = {name: k for k, name in enumerate(PAIR_FEATURES)}
_GLOBAL = {name: k for k, name in enumerate(GLOBAL_FEATURES)}


@dataclasses.dataclass(frozen=True)
class Observation:
    """
    Features of one state; the only input of a policy.
    """

    # Shape (n_nodes, len(NODE_FEATURES)).
    node: np.ndarray
    # Shape (n_nodes, n_nodes, len(PAIR_FEATURES)); the diagonal is zero.
    pair: np.ndarray
    # Shape (len(GLOBAL_FEATURES),).
    glob: np.ndarray


def schema_hash() -> str:
    """
    Hash the feature names, so checkpoints refuse a changed feature layout.

    :return: short hex digest
    """
    text = "|".join(NODE_FEATURES + PAIR_FEATURES + GLOBAL_FEATURES)
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def _signed_log1p(value: float) -> float:
    """
    Compress a statistic while keeping its sign.

    :param value: statistic
    :return: `sign(value) * log(1 + |value|)`
    """
    return math.copysign(math.log1p(abs(value)), value)


def _neglog10_p(p_value: float) -> float:
    """
    Map a p-value to [0, 1]: -log10(p) / 10, capped at 1.

    :param p_value: p-value
    :return: scaled evidence against the null
    """
    return min(-math.log10(max(p_value, 1e-300)), 10.0) / 10.0


def _add_free_features(
    evidence: ttools.Evidence, node: np.ndarray, pair: np.ndarray
) -> None:
    """
    Fill the features computed from the free correlation matrix.

    :param evidence: evidence
    :param node: node features, filled in place
    :param pair: pair features, filled in place
    """
    n_nodes = evidence.n_nodes
    corr = evidence.corr.copy()
    np.fill_diagonal(corr, 0.0)
    abs_r = np.abs(corr)
    pair[:, :, _PAIR["abs_r"]] = abs_r
    pair[:, :, _PAIR["r"]] = corr
    r_clipped = np.clip(corr, -0.999999, 0.999999)
    fisher_z = np.arctanh(r_clipped) * math.sqrt(evidence.n_train - 3)
    pair[:, :, _PAIR["fisher_z"]] = np.tanh(fisher_z / 10.0)
    # max_k |r_ik r_jk| over k != i, j (the diagonal of `abs_r` is zero).
    triad = (abs_r[:, None, :] * abs_r[None, :, :]).max(axis=2)
    np.fill_diagonal(triad, 0.0)
    pair[:, :, _PAIR["triad_abs_r"]] = triad
    node[:, _NODE["mean_abs_r"]] = abs_r.sum(axis=1) / max(1, n_nodes - 1)
    node[:, _NODE["max_abs_r"]] = abs_r.max(axis=1)


def _add_tool_features(
    evidence: ttools.Evidence, node: np.ndarray, pair: np.ndarray
) -> None:
    """
    Fill the features computed from tool results.

    :param evidence: evidence
    :param node: node features, filled in place
    :param pair: pair features, filled in place
    """
    n_nodes = evidence.n_nodes
    n_ci_1 = np.zeros((n_nodes, n_nodes))
    n_ci_2 = np.zeros((n_nodes, n_nodes))
    min_pcorr = np.ones((n_nodes, n_nodes))
    best_p = np.zeros((n_nodes, n_nodes))
    best_size = np.zeros((n_nodes, n_nodes))
    lingam_tested = np.zeros((n_nodes, n_nodes))
    for action, result in evidence.results.items():
        args = action.args
        if action.kind == ttools.CI_TEST:
            i, j, size = args[0], args[1], len(args) - 2
            counts = n_ci_1 if size == 1 else n_ci_2
            counts[i, j] += 1
            counts[j, i] += 1
            min_pcorr[i, j] = min_pcorr[j, i] = min(
                min_pcorr[i, j], abs(result["pcorr"])
            )
            if result["p"] > best_p[i, j]:
                best_p[i, j] = best_p[j, i] = result["p"]
                best_size[i, j] = best_size[j, i] = size
        elif action.kind == ttools.REGRESS:
            target, preds = args[0], args[1:]
            node[target, _NODE["regress_target_mask"]] = 1.0
            node[target, _NODE["best_r2_as_target"]] = max(
                node[target, _NODE["best_r2_as_target"]], result["r2"]
            )
            if len(preds) == 1:
                # Regressing `target` on `cause`: evidence about cause -> target.
                cause = preds[0]
                fwd = pair[cause, target]
                fwd[_PAIR["regress_fwd_mask"]] = 1.0
                fwd[_PAIR["regress_fwd_t"]] = math.tanh(
                    result[f"t_{cause}"] / 10.0
                )
                fwd[_PAIR["regress_fwd_dep"]] = result["resid_dep"]
                fwd[_PAIR["regress_fwd_ad"]] = (
                    math.log1p(result["resid_ad"]) / 5.0
                )
                bwd = pair[target, cause]
                bwd[_PAIR["regress_bwd_mask"]] = 1.0
                bwd[_PAIR["regress_bwd_dep"]] = result["resid_dep"]
        elif action.kind == ttools.LINGAM_DIR:
            i, j = args
            value = math.tanh(20.0 * result["stat"])
            pair[i, j, _PAIR["lingam_mask"]] = 1.0
            pair[j, i, _PAIR["lingam_mask"]] = 1.0
            pair[i, j, _PAIR["lingam_dir"]] = value
            pair[j, i, _PAIR["lingam_dir"]] = -value
            lingam_tested[i, j] = lingam_tested[j, i] = 1.0
        elif action.kind == ttools.NONGAUSS:
            (i,) = args
            node[i, _NODE["nongauss_mask"]] = 1.0
            node[i, _NODE["skew"]] = _signed_log1p(result["skew"])
            node[i, _NODE["kurt"]] = _signed_log1p(result["kurt"])
            node[i, _NODE["jb_neglog_p"]] = _neglog10_p(result["jb_p"])
        elif action.kind == ttools.VARIANCE:
            (i,) = args
            node[i, _NODE["variance_mask"]] = 1.0
            node[i, _NODE["log_var"]] = float(np.clip(result["log_var"], -5, 5))
    # CI summaries.
    n_others = max(1, n_nodes - 2)
    n_pairs_of_others = max(1, n_others * (n_others - 1) // 2)
    tested = (n_ci_1 + n_ci_2) > 0
    pair[:, :, _PAIR["ci_mask"]] = tested
    pair[:, :, _PAIR["ci_cover_1"]] = n_ci_1 / n_others
    pair[:, :, _PAIR["ci_cover_2"]] = n_ci_2 / n_pairs_of_others
    pair[:, :, _PAIR["min_abs_pcorr"]] = np.where(tested, min_pcorr, 0.0)
    pair[:, :, _PAIR["indep_p01"]] = best_p > 0.01
    pair[:, :, _PAIR["indep_p05"]] = best_p > 0.05
    pair[:, :, _PAIR["best_sep_size"]] = (
        np.where(best_p > 0.01, best_size, 0) / 2
    )
    # Prototype direction statistic: dep(j, i) - dep(i, j) > 0 favours i -> j.
    both = (pair[:, :, _PAIR["regress_fwd_mask"]] > 0) & (
        pair[:, :, _PAIR["regress_bwd_mask"]] > 0
    )
    diff = (
        pair[:, :, _PAIR["regress_bwd_dep"]]
        - pair[:, :, _PAIR["regress_fwd_dep"]]
    )
    pair[:, :, _PAIR["proto_dir"]] = np.where(both, diff, 0.0)
    # Coverage per node.
    node[:, _NODE["frac_pairs_ci_tested"]] = tested.sum(axis=1) / max(
        1, n_nodes - 1
    )
    node[:, _NODE["frac_pairs_lingam_tested"]] = lingam_tested.sum(axis=1) / max(
        1, n_nodes - 1
    )


def featurize(evidence: ttools.Evidence) -> Observation:
    """
    Compute the features of the current state.

    :param evidence: evidence of the episode so far
    :return: node, pair and global features (float32)
    """
    n_nodes = evidence.n_nodes
    node = np.zeros((n_nodes, len(NODE_FEATURES)))
    pair = np.zeros((n_nodes, n_nodes, len(PAIR_FEATURES)))
    glob = np.zeros(len(GLOBAL_FEATURES))
    _add_free_features(evidence, node, pair)
    _add_tool_features(evidence, node, pair)
    for k in range(n_nodes):
        pair[k, k, :] = 0.0
    budget = max(1, evidence.budget)
    tested_nodes = node[:, _NODE["nongauss_mask"]] > 0
    upper = np.triu_indices(n_nodes, k=1)
    glob[_GLOBAL["log_n_nodes"]] = math.log(n_nodes) / math.log(16)
    glob[_GLOBAL["log_n_train"]] = math.log(evidence.n_train) / math.log(10000)
    glob[_GLOBAL["budget_left_frac"]] = evidence.budget_left / budget
    glob[_GLOBAL["calls_used_frac"]] = evidence.calls_used / budget
    glob[_GLOBAL["log_budget"]] = math.log1p(evidence.budget) / math.log(1000)
    glob[_GLOBAL["frac_pairs_ci_tested"]] = pair[:, :, _PAIR["ci_mask"]][
        upper
    ].mean()
    glob[_GLOBAL["frac_nodes_nongauss_tested"]] = tested_nodes.mean()
    if tested_nodes.any():
        glob[_GLOBAL["mean_abs_skew"]] = np.abs(
            node[tested_nodes, _NODE["skew"]]
        ).mean()
        glob[_GLOBAL["mean_abs_kurt"]] = np.abs(
            node[tested_nodes, _NODE["kurt"]]
        ).mean()
    glob[_GLOBAL["mean_abs_r"]] = pair[:, :, _PAIR["abs_r"]][upper].mean()
    obs = Observation(
        node=node.astype(np.float32),
        pair=pair.astype(np.float32),
        glob=glob.astype(np.float32),
    )
    return obs


def permute_observation(obs: Observation, perm: List[int]) -> Observation:
    """
    Relabel the variables of an observation.

    New variable `k` is old variable `perm[k]`.

    :param obs: observation
    :param perm: permutation of `range(n_nodes)`
    :return: relabelled observation
    """
    perm_arr = np.asarray(perm)
    permuted = Observation(
        node=obs.node[perm_arr],
        pair=obs.pair[np.ix_(perm_arr, perm_arr)],
        glob=obs.glob,
    )
    return permuted
