"""
Score a discovered graph against the hidden graph: structure and prediction.

Conventions:
- A DAG is a binary adjacency `B` with `B[i, j] = 1` meaning i -> j (the
  prototype's convention)
- A PDAG is a binary adjacency `A` where `A[i, j] = A[j, i] = 1` means the
  undirected edge i - j
- Marks summarize each unordered pair i < j as one of: none, i -> j, j -> i,
  undirected, bidirected; SHD counts the pairs whose marks differ, so a
  reversed edge costs 1 (as in the prototype and causal-learn)

The combined score follows the spec and GP's draft paper:
`score = -lam * SHD_target / max(1, |E*|) - (1 - lam) * (1 - pred_frac)`,
where `SHD_target` compares against what the data can identify (the DAG, its
equivalence class, or the non-Gaussian pattern) and `pred_frac` is the held-out
log-likelihood gain of the discovered graph over the empty graph, relative to
the true graph's gain. The true graph scores 0 and the empty graph -1.

Import as:

import tukey_scoring as tscore
"""

import dataclasses
import itertools
import logging
import types
from typing import Any, Dict, List, Tuple

import numpy as np

import helpers.hdbg as hdbg
import research.agentic_data_science.RL_for_auto_EDA_prototype as rlproto

_LOG = logging.getLogger(__name__)

# Mark codes for an unordered pair i < j.
MARK_NONE = 0
MARK_FORWARD = 1
MARK_BACKWARD = 2
MARK_UNDIRECTED = 3
MARK_BIDIRECTED = 4

# Target classes: what the data can identify for an instance.
TARGET_DAG = "dag"
TARGET_CPDAG = "cpdag"
TARGET_NGDAG = "ngdag_pattern"

# Minimum held-out log-likelihood gain of the true graph (nats per sample) for
# the prediction term to be informative.
PRED_GAIN_MIN = 0.02


# #############################################################################
# Graph conversions
# #############################################################################


def edges_to_adjacency(
    edges: List[Tuple[int, int]], n_nodes: int
) -> Tuple[np.ndarray, int]:
    """
    Convert an edge list into a binary adjacency, as the prototype reads it.

    Self-loops, out-of-range indices and duplicates are dropped, as in the
    prototype's `read_graph()`.

    :param edges: `[(parent, child), ...]`
    :param n_nodes: number of nodes
    :return: adjacency `B` and the number of dropped entries
    """
    B = np.zeros((n_nodes, n_nodes), dtype=int)
    n_dropped = 0
    for parent, child in edges:
        valid = 0 <= parent < n_nodes and 0 <= child < n_nodes
        if not valid or parent == child or B[parent, child]:
            n_dropped += 1
            continue
        B[parent, child] = 1
    return B, n_dropped


def is_dag(B: np.ndarray) -> bool:
    """
    Check whether a directed adjacency has no directed cycle.

    :param B: directed adjacency (`B[i, j] = 1` means i -> j)
    :return: True if acyclic
    """
    # Kahn's algorithm: repeatedly remove nodes with no incoming edge.
    in_degree = B.sum(axis=0).astype(int)
    stack = list(np.flatnonzero(in_degree == 0))
    n_removed = 0
    remaining = B.astype(int).copy()
    while stack:
        node = stack.pop()
        n_removed += 1
        for child in np.flatnonzero(remaining[node]):
            remaining[node, child] = 0
            in_degree[child] -= 1
            if in_degree[child] == 0:
                stack.append(child)
    acyclic = n_removed == B.shape[0]
    return acyclic


def marks_from_pdag(A: np.ndarray) -> np.ndarray:
    """
    Summarize a (P)DAG as one mark per unordered pair.

    :param A: adjacency; `A[i, j] = A[j, i] = 1` is read as undirected
    :return: integer matrix whose upper triangle holds the mark of each pair
        i < j; the rest is 0
    """
    forward = np.triu(A, 1)
    backward = np.triu(A.T, 1)
    marks = np.zeros(A.shape, dtype=int)
    marks[(forward == 1) & (backward == 0)] = MARK_FORWARD
    marks[(forward == 0) & (backward == 1)] = MARK_BACKWARD
    marks[(forward == 1) & (backward == 1)] = MARK_UNDIRECTED
    return marks


def shd_marks(marks1: np.ndarray, marks2: np.ndarray) -> int:
    """
    Count the unordered pairs whose marks differ.

    :param marks1: marks of the first graph
    :param marks2: marks of the second graph
    :return: structural Hamming distance
    """
    shd = int(np.triu(marks1 != marks2, 1).sum())
    return shd


def _orient(A: np.ndarray, src: int, dst: int) -> None:
    """
    Turn the undirected edge src - dst into src -> dst, in place.

    :param A: PDAG adjacency
    :param src: tail of the oriented edge
    :param dst: head of the oriented edge
    """
    A[dst, src] = 0


def _apply_meek_rules(A: np.ndarray) -> np.ndarray:
    """
    Orient undirected edges with Meek's rules R1-R3 until nothing changes.

    R1: a -> u - v, a and v non-adjacent => u -> v.
    R2: u -> w -> v and u - v => u -> v.
    R3: u - c, u - d, c -> v, d -> v, c and d non-adjacent, u - v => u -> v.
    For a PDAG obtained from a DAG's skeleton and v-structures, R1-R3 give its
    equivalence class (Meek 1995); R4 is needed only with background knowledge.

    :param A: PDAG adjacency, modified in place
    :return: the same array
    """
    adjacent = (A + A.T) > 0
    changed = True
    while changed:
        changed = False
        directed = (A == 1) & (A.T == 0)
        undirected = (A == 1) & (A.T == 1)
        for u, v in zip(*np.nonzero(np.triu(undirected, 1))):
            for src, dst in ((u, v), (v, u)):
                # R1: some a -> src with a not adjacent to dst.
                parents = np.flatnonzero(directed[:, src])
                rule1 = any(not adjacent[a, dst] for a in parents if a != dst)
                # R2: src -> w -> dst for some w.
                rule2 = bool(np.any(directed[src] & directed[:, dst]))
                # R3: two non-adjacent c, d with src - c, src - d, c -> dst,
                # d -> dst.
                cands = np.flatnonzero(undirected[src] & directed[:, dst])
                rule3 = any(
                    not adjacent[c, d]
                    for c, d in itertools.combinations(cands, 2)
                )
                if rule1 or rule2 or rule3:
                    _orient(A, src, dst)
                    changed = True
                    break
            if changed:
                break
    return A


def cpdag_from_dag(B: np.ndarray) -> np.ndarray:
    """
    Compute the equivalence class (CPDAG) of a DAG.

    Keeps the skeleton, orients the v-structures, then applies Meek's rules.
    Cross-checked against causal-learn's `dag2cpdag()` and brute-force class
    enumeration in the unit tests.

    :param B: DAG adjacency
    :return: PDAG adjacency of the CPDAG
    """
    hdbg.dassert(is_dag(B), "cpdag_from_dag needs a DAG")
    skeleton = ((B + B.T) > 0).astype(int)
    A = skeleton.copy()
    # Orient every v-structure i -> k <- j with i, j non-adjacent.
    for k in range(B.shape[0]):
        parents = np.flatnonzero(B[:, k])
        for i, j in itertools.combinations(parents, 2):
            if not skeleton[i, j]:
                _orient(A, i, k)
                _orient(A, j, k)
    A = _apply_meek_rules(A)
    return A


def ngdag_pattern_from_dag(B: np.ndarray, nongauss: np.ndarray) -> np.ndarray:
    """
    Compute what is identifiable when some noise terms are non-Gaussian.

    Hoyer et al. 2008: start from the equivalence class and orient every edge
    touching a non-Gaussian node, then propagate. Implemented with their trick:
    add two parentless parents to each non-Gaussian node, take the CPDAG, drop
    the added nodes.

    :param B: DAG adjacency
    :param nongauss: boolean per node, True if its noise is non-Gaussian
    :return: PDAG adjacency of the pattern
    """
    d = B.shape[0]
    ng_nodes = np.flatnonzero(nongauss)
    n_total = d + 2 * len(ng_nodes)
    B_aug = np.zeros((n_total, n_total), dtype=int)
    B_aug[:d, :d] = B
    for idx, node in enumerate(ng_nodes):
        B_aug[d + 2 * idx, node] = 1
        B_aug[d + 2 * idx + 1, node] = 1
    pattern = cpdag_from_dag(B_aug)[:d, :d]
    return pattern


def dag_extension(A: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Find a DAG consistent with a PDAG (Dor and Tarsi 1992).

    Repeatedly picks a node with no outgoing directed edge whose undirected
    neighbours are adjacent to all its other neighbours, orients its undirected
    edges into it, and removes it.

    :param A: PDAG adjacency
    :return: a consistent DAG, and False if the PDAG admits no extension (the
        returned matrix is then the PDAG's directed part)
    """
    d = A.shape[0]
    work = A.copy()
    B = ((A == 1) & (A.T == 0)).astype(int)
    alive = np.ones(d, dtype=bool)
    for _ in range(d):
        found = False
        for x in np.flatnonzero(alive):
            directed_out = (work[x] == 1) & (work[:, x] == 0) & alive
            if directed_out.any():
                continue
            undirected = np.flatnonzero(
                (work[x] == 1) & (work[:, x] == 1) & alive
            )
            neighbours = np.flatnonzero(((work[x] + work[:, x]) > 0) & alive)
            ok = all(
                ((work[y, z] + work[z, y]) > 0) or y == z
                for y in undirected
                for z in neighbours
            )
            if not ok:
                continue
            # Orient every undirected edge y - x as y -> x, then remove x.
            for y in undirected:
                B[y, x] = 1
            alive[x] = False
            found = True
            break
        if not found:
            return B, False
    return B, True


def target_marks(
    B: np.ndarray, target_class: str, nongauss: np.ndarray
) -> np.ndarray:
    """
    Compute the marks of a DAG under an identifiability class.

    :param B: DAG adjacency
    :param target_class: `TARGET_DAG`, `TARGET_CPDAG` or `TARGET_NGDAG`
    :param nongauss: boolean per node (used by `TARGET_NGDAG`)
    :return: marks matrix
    """
    if target_class == TARGET_DAG:
        A = B
    elif target_class == TARGET_CPDAG:
        A = cpdag_from_dag(B)
    elif target_class == TARGET_NGDAG:
        A = ngdag_pattern_from_dag(B, nongauss)
    else:
        raise ValueError(f"Unknown target_class='{target_class}'")
    marks = marks_from_pdag(A)
    return marks


# #############################################################################
# Held-out prediction
# #############################################################################


def get_moments(x: np.ndarray) -> np.ndarray:
    """
    Compute raw second moments of the data with an intercept column.

    :param x: data, shape (n_samples, n_nodes)
    :return: `Z.T @ Z / n` with `Z = [1, x]`, shape (n_nodes + 1, n_nodes + 1)
    """
    n = x.shape[0]
    z = np.hstack([np.ones((n, 1)), x])
    moments = z.T @ z / n
    return moments


def pred_ll(
    B: np.ndarray, moments_train: np.ndarray, moments_test: np.ndarray
) -> Tuple[float, bool]:
    """
    Compute the held-out Gaussian log-likelihood of a DAG, per sample.

    Each node is regressed on its parents (OLS with intercept) on the training
    rows; the residual variance is the MLE `RSS / n`. The mean log-density of
    the test rows is summed over nodes. Markov-equivalent DAGs fit the same
    Gaussian model, so they get the same value.

    :param B: DAG adjacency
    :param moments_train: `get_moments()` of the training rows
    :param moments_test: `get_moments()` of the test rows
    :return: log-likelihood in nats per sample, and whether a residual
        variance had to be floored (near-deterministic node)
    """
    total = 0.0
    degenerate = False
    for j in range(B.shape[0]):
        # Index 0 is the intercept; node k is at index k + 1.
        idx = [0] + [p + 1 for p in np.flatnonzero(B[:, j])]
        target = j + 1
        coef = np.linalg.solve(
            moments_train[np.ix_(idx, idx)], moments_train[idx, target]
        )
        sigma2 = (
            moments_train[target, target] - coef @ moments_train[idx, target]
        )
        floor = 1e-10 * max(1.0, moments_train[target, target])
        if sigma2 < floor:
            sigma2 = floor
            degenerate = True
        # Mean squared residual on the test rows, from test moments.
        test_mse = (
            moments_test[target, target]
            - 2 * coef @ moments_test[idx, target]
            + coef @ moments_test[np.ix_(idx, idx)] @ coef
        )
        total += -0.5 * np.log(2 * np.pi * sigma2) - test_mse / (2 * sigma2)
    return float(total), degenerate


# #############################################################################
# Score
# #############################################################################


@dataclasses.dataclass(frozen=True)
class Truth:
    """
    Everything the scorer needs about one instance; never shown to an agent.
    """

    # True DAG adjacency.
    B: np.ndarray
    # Identifiability class of the instance.
    target_class: str
    # Boolean per node: True if its noise is non-Gaussian.
    nongauss: np.ndarray
    # Raw moments of the training and test rows (see `get_moments()`).
    moments_train: np.ndarray
    moments_test: np.ndarray
    # Data, used only for the prototype's legacy `oos_r2_mean` key.
    x_train: np.ndarray
    x_test: np.ndarray
    # Held-out log-likelihoods of the empty and the true graph (cached).
    ll_empty: float
    ll_true: float


def make_truth(
    B: np.ndarray,
    target_class: str,
    nongauss: np.ndarray,
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> Truth:
    """
    Build the scorer's view of an instance, caching the reference likelihoods.

    :param B: true DAG adjacency
    :param target_class: identifiability class of the instance
    :param nongauss: boolean per node, True if its noise is non-Gaussian
    :param x_train: training rows, shape (n_train, n_nodes)
    :param x_test: test rows, shape (n_test, n_nodes)
    :return: `Truth` with moments and the empty/true log-likelihoods
    """
    moments_train = get_moments(x_train)
    moments_test = get_moments(x_test)
    ll_empty, _ = pred_ll(np.zeros_like(B), moments_train, moments_test)
    ll_true, _ = pred_ll(B, moments_train, moments_test)
    truth = Truth(
        B=B,
        target_class=target_class,
        nongauss=nongauss,
        moments_train=moments_train,
        moments_test=moments_test,
        x_train=x_train,
        x_test=x_test,
        ll_empty=ll_empty,
        ll_true=ll_true,
    )
    return truth


def _get_prediction_terms(B_hat: np.ndarray, truth: Truth) -> Dict[str, Any]:
    """
    Compute the held-out log-likelihood terms of a DAG.

    :param B_hat: discovered DAG
    :param truth: instance truth
    :return: `pred_ll`, `pred_gain`, `pred_frac`, `prediction_informative`,
        `degenerate`
    """
    ll_hat, degenerate = pred_ll(B_hat, truth.moments_train, truth.moments_test)
    gain_hat = ll_hat - truth.ll_empty
    gain_true = truth.ll_true - truth.ll_empty
    informative = gain_true >= PRED_GAIN_MIN
    pred_frac = (
        float(np.clip(gain_hat / gain_true, -1.0, 1.5)) if informative else 0.0
    )
    terms = {
        "pred_ll": ll_hat,
        "pred_gain": gain_hat,
        "pred_frac": pred_frac,
        "prediction_informative": bool(informative),
        "degenerate": degenerate,
    }
    return terms


def _get_skeleton_scores(B_hat: np.ndarray, B: np.ndarray) -> Dict[str, float]:
    """
    Compute skeleton F1 and orientation accuracy on correctly found edges.

    :param B_hat: discovered graph (directed adjacency)
    :param B: true DAG
    :return: `skeleton_f1`, `orientation_accuracy` (NaN if no edge is shared)
    """
    skel_hat = np.triu((B_hat + B_hat.T) > 0, 1)
    skel = np.triu((B + B.T) > 0, 1)
    tp = int((skel_hat & skel).sum())
    denom = int(skel_hat.sum() + skel.sum())
    skeleton_f1 = 2 * tp / denom if denom else 1.0
    shared = np.argwhere(skel_hat & skel)
    n_correct = sum(
        int(B_hat[i, j] == B[i, j] and B_hat[j, i] == B[j, i]) for i, j in shared
    )
    orientation = n_correct / len(shared) if len(shared) else float("nan")
    scores = {"skeleton_f1": skeleton_f1, "orientation_accuracy": orientation}
    return scores


def _get_legacy_scores(B_hat: np.ndarray, truth: Truth) -> Dict[str, float]:
    """
    Compute the prototype's score keys with the prototype's own `score()`.

    :param B_hat: discovered graph (directed adjacency)
    :param truth: instance truth
    :return: `shd`, `tpr`, `fdr`, `f1`, `oos_r2_mean` as the prototype defines
    """
    env_like = types.SimpleNamespace(
        n_nodes=truth.B.shape[0],
        B_true=truth.B,
        X_train=truth.x_train,
        X_test=truth.x_test,
    )
    legacy = rlproto.score(B_hat, env_like)
    return legacy


def score_pdag(
    A_hat: np.ndarray, truth: Truth, *, lam: float = 0.5
) -> Dict[str, Any]:
    """
    Score a PDAG output (e.g., PC or GES) against the instance truth.

    Marks are compared as emitted: an undirected edge matches only an
    undirected edge of the target. The prediction term uses any DAG extension
    (all extensions of a valid PDAG are Markov equivalent, so they agree).

    :param A_hat: discovered PDAG adjacency
    :param truth: instance truth
    :param lam: weight of the structural term
        - Default: 0.5
    :return: score dictionary (same keys as `score_graph()` minus the legacy
        prototype keys) plus `extendable`
    """
    n_true_edges = int(truth.B.sum())
    marks_hat = marks_from_pdag(A_hat)
    marks_true_target = target_marks(truth.B, truth.target_class, truth.nongauss)
    B_ext, extendable = dag_extension(A_hat)
    scores: Dict[str, Any] = {
        "is_dag": False,
        "extendable": extendable,
        "target_class": truth.target_class,
        "n_true_edges": n_true_edges,
        "shd_dag": shd_marks(marks_hat, marks_from_pdag(truth.B)),
        "shd_cpdag": shd_marks(
            marks_hat, marks_from_pdag(cpdag_from_dag(truth.B))
        ),
        "shd_target": shd_marks(marks_hat, marks_true_target),
    }
    scores.update(_get_skeleton_scores(B_ext, truth.B))
    if extendable:
        scores.update(_get_prediction_terms(B_ext, truth))
        shd_norm = scores["shd_target"] / max(1, n_true_edges)
        if scores["prediction_informative"]:
            score = -lam * shd_norm - (1 - lam) * (1 - scores["pred_frac"])
        else:
            score = -shd_norm
        # Adding 0.0 turns a negative zero (e.g., -0.5 * 0) into 0.0.
        scores["score"] = float(score) + 0.0
    else:
        scores["score"] = float("nan")
    scores["lam"] = lam
    return scores


def score_graph(
    B_hat: np.ndarray,
    truth: Truth,
    *,
    lam: float = 0.5,
    legacy: bool = True,
) -> Dict[str, Any]:
    """
    Score a discovered directed graph against the instance truth.

    Cyclic graphs get structural scores only (`is_dag` False): the
    log-likelihood is not a valid likelihood for them and the combined score is
    left undefined (NaN).

    :param B_hat: discovered graph, directed adjacency
    :param truth: instance truth
    :param lam: weight of the structural term
        - Default: 0.5
    :param legacy: also compute the prototype's keys (slower; off in training)
        - Default: True
    :return: score dictionary
    """
    hdbg.dassert_eq(B_hat.shape, truth.B.shape)
    n_true_edges = int(truth.B.sum())
    acyclic = is_dag(B_hat)
    marks_true_dag = marks_from_pdag(truth.B)
    marks_true_cpdag = marks_from_pdag(cpdag_from_dag(truth.B))
    marks_true_target = target_marks(truth.B, truth.target_class, truth.nongauss)
    marks_hat_dag = marks_from_pdag(B_hat)
    if acyclic:
        marks_hat_cpdag = marks_from_pdag(cpdag_from_dag(B_hat))
        marks_hat_target = target_marks(
            B_hat, truth.target_class, truth.nongauss
        )
    else:
        # No equivalence-class credit for a cyclic graph.
        marks_hat_cpdag = marks_hat_dag
        marks_hat_target = marks_hat_dag
    scores: Dict[str, Any] = {
        "is_dag": acyclic,
        "target_class": truth.target_class,
        "n_true_edges": n_true_edges,
        "shd_dag": shd_marks(marks_hat_dag, marks_true_dag),
        "shd_cpdag": shd_marks(marks_hat_cpdag, marks_true_cpdag),
        "shd_target": shd_marks(marks_hat_target, marks_true_target),
    }
    scores.update(_get_skeleton_scores(B_hat, truth.B))
    if acyclic:
        scores.update(_get_prediction_terms(B_hat, truth))
        shd_norm = scores["shd_target"] / max(1, n_true_edges)
        if scores["prediction_informative"]:
            score = -lam * shd_norm - (1 - lam) * (1 - scores["pred_frac"])
        else:
            score = -shd_norm
        # Adding 0.0 turns a negative zero (e.g., -0.5 * 0) into 0.0.
        scores["score"] = float(score) + 0.0
    else:
        scores["score"] = float("nan")
    scores["lam"] = lam
    if legacy:
        scores.update(_get_legacy_scores(B_hat, truth))
    return scores
