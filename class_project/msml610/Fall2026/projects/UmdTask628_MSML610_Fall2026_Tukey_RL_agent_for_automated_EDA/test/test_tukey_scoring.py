"""
Test the Tukey scorer: marks, SHD variants, equivalence classes, prediction.

This is integration gate G2. Expected values are written out by hand for small
graphs; property tests compare against brute-force class enumeration,
causal-learn and the prototype. Graphs inside one equivalence class are
expected to tie wherever the target is the class.

Import as:

import test.test_tukey_scoring as ttetusco
"""

import itertools
import logging
from typing import Dict, List, Tuple

import causallearn.graph.Dag as cgrdag
import causallearn.graph.GraphNode as cgrgnod
import causallearn.graph.SHD as cgrshd
import causallearn.utils.DAG2CPDAG as cudcpdag
import numpy as np

import helpers.hunit_test as hunitest
import research.agentic_data_science.RL_for_auto_EDA_prototype as rlproto
import tukey_scoring as tscore

_LOG = logging.getLogger(__name__)


# #############################################################################
# Fixtures
# #############################################################################


def _adjacency(n_nodes: int, edges: List[Tuple[int, int]]) -> np.ndarray:
    """
    Build a directed adjacency from an edge list.

    :param n_nodes: number of nodes
    :param edges: `[(parent, child), ...]`
    :return: adjacency with `B[parent, child] = 1`
    """
    B = np.zeros((n_nodes, n_nodes), dtype=int)
    for parent, child in edges:
        B[parent, child] = 1
    return B


def _topological_order(B: np.ndarray) -> List[int]:
    """
    Return a topological order of a DAG.

    :param B: DAG adjacency
    :return: node order with parents before children
    """
    order: List[int] = []
    remaining = set(range(B.shape[0]))
    while remaining:
        for j in sorted(remaining):
            if all(p in order for p in np.flatnonzero(B[:, j])):
                order.append(j)
                remaining.remove(j)
                break
    return order


def _simulate(
    B: np.ndarray, *, seed: int, n_samples: int = 6000, noise: str = "gaussian"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample a linear SEM on `B` with weight 0.8 and split it 2:1.

    :param B: DAG adjacency
    :param seed: random seed
    :param n_samples: number of rows
    :param noise: "gaussian" or "exponential" (centred)
    :return: training rows and test rows
    """
    rng = np.random.default_rng(seed)
    d = B.shape[0]
    x = np.zeros((n_samples, d))
    order = _topological_order(B)
    for j in order:
        if noise == "gaussian":
            e = rng.normal(size=n_samples)
        else:
            e = rng.exponential(size=n_samples) - 1.0
        x[:, j] = x @ (0.8 * B[:, j]) + e
    cut = 2 * n_samples // 3
    return x[:cut], x[cut:]


def _truth(
    B: np.ndarray,
    target_class: str,
    *,
    seed: int = 0,
    noise: str = "gaussian",
    nongauss: np.ndarray = np.zeros(0, dtype=bool),
) -> tscore.Truth:
    """
    Build a `Truth` for a DAG with simulated data.

    :param B: true DAG
    :param target_class: identifiability class
    :param seed: data seed
    :param noise: noise family for the simulation
    :param nongauss: non-Gaussian mask; empty means all Gaussian
    :return: instance truth
    """
    x_train, x_test = _simulate(B, seed=seed, noise=noise)
    if nongauss.size == 0:
        nongauss = np.zeros(B.shape[0], dtype=bool)
    truth = tscore.make_truth(B, target_class, nongauss, x_train, x_test)
    return truth


def _random_dag(
    rng: np.random.Generator, n_nodes: int, p_edge: float
) -> np.ndarray:
    """
    Sample a random DAG by orienting a random graph along a random order.

    :param rng: random generator
    :param n_nodes: number of nodes
    :param p_edge: probability of each edge
    :return: DAG adjacency
    """
    order = rng.permutation(n_nodes)
    B = np.zeros((n_nodes, n_nodes), dtype=int)
    for a, b in itertools.combinations(range(n_nodes), 2):
        if rng.random() < p_edge:
            B[order[a], order[b]] = 1
    return B


def _v_structures(B: np.ndarray) -> set:
    """
    List the v-structures i -> k <- j (i, j non-adjacent) of a DAG.

    :param B: DAG adjacency
    :return: set of `(min(i, j), max(i, j), k)`
    """
    skeleton = (B + B.T) > 0
    vstructs = set()
    for k in range(B.shape[0]):
        for i, j in itertools.combinations(np.flatnonzero(B[:, k]), 2):
            if not skeleton[i, j]:
                vstructs.add((min(i, j), max(i, j), k))
    return vstructs


def _brute_force_cpdag(B: np.ndarray) -> np.ndarray:
    """
    Compute the CPDAG by enumerating every DAG with the same skeleton.

    Two DAGs are Markov equivalent iff they share the skeleton and the
    v-structures (Verma and Pearl 1990); an edge is directed in the CPDAG iff
    every member orients it the same way.

    :param B: DAG adjacency
    :return: PDAG adjacency
    """
    edges = [tuple(e) for e in np.argwhere(B)]
    vstructs = _v_structures(B)
    members = []
    for flips in itertools.product([False, True], repeat=len(edges)):
        cand = np.zeros_like(B)
        for (i, j), flip in zip(edges, flips):
            if flip:
                cand[j, i] = 1
            else:
                cand[i, j] = 1
        if tscore.is_dag(cand) and _v_structures(cand) == vstructs:
            members.append(cand)
    A = np.zeros_like(B)
    for i, j in edges:
        if all(m[i, j] for m in members):
            A[i, j] = 1
        elif all(m[j, i] for m in members):
            A[j, i] = 1
        else:
            A[i, j] = A[j, i] = 1
    return A


def _causallearn_cpdag(B: np.ndarray) -> np.ndarray:
    """
    Compute the CPDAG with causal-learn's `dag2cpdag()`.

    :param B: DAG adjacency
    :return: PDAG adjacency in Tukey's convention
    """
    nodes = [cgrgnod.GraphNode(f"x{i}") for i in range(B.shape[0])]
    dag = cgrdag.Dag(nodes)
    for i, j in np.argwhere(B):
        dag.add_directed_edge(nodes[i], nodes[j])
    graph = cudcpdag.dag2cpdag(dag).graph
    # causal-learn: graph[j, i] = 1 and graph[i, j] = -1 means i -> j;
    # graph[i, j] = graph[j, i] = -1 means i - j.
    A = np.zeros_like(B)
    for i, j in itertools.permutations(range(B.shape[0]), 2):
        if graph[i, j] == -1 and graph[j, i] == 1:
            A[i, j] = 1
        elif graph[i, j] == -1 and graph[j, i] == -1:
            A[i, j] = 1
    return A


def _summary(scores: Dict) -> str:
    """
    Format the structural keys of a score dictionary.

    :param scores: output of `score_graph()`
    :return: compact string for comparison
    """
    keys = ["is_dag", "shd_dag", "shd_cpdag", "shd_target"]
    summary = " ".join(f"{k}={scores[k]}" for k in keys)
    return summary


# #############################################################################
# Test_score_graph
# #############################################################################


class Test_score_graph(hunitest.TestCase):
    """
    Test `tukey_scoring.score_graph()` on hand-checked small graphs.
    """

    def helper(
        self,
        n_nodes: int,
        true_edges: List[Tuple[int, int]],
        hat_edges: List[Tuple[int, int]],
        target_class: str,
        expected: str,
    ) -> Dict:
        """
        Score `hat_edges` against `true_edges` and compare structural keys.

        :param n_nodes: number of nodes
        :param true_edges: true DAG edges
        :param hat_edges: discovered edges
        :param target_class: identifiability class
        :param expected: expected `_summary()` string
        :return: the score dictionary, for extra checks
        """
        # Prepare inputs.
        truth = _truth(_adjacency(n_nodes, true_edges), target_class)
        B_hat = _adjacency(n_nodes, hat_edges)
        # Run test.
        scores = tscore.score_graph(B_hat, truth)
        # Check outputs.
        self.assert_equal(_summary(scores), expected)
        return scores

    def test1(self) -> None:
        """
        Test that the true complete DAG scores perfectly.
        """
        edges = [(0, 1), (1, 2), (0, 2)]
        expected = "is_dag=True shd_dag=0 shd_cpdag=0 shd_target=0"
        scores = self.helper(3, edges, edges, tscore.TARGET_CPDAG, expected)
        self.assert_equal(
            str([scores["pred_frac"], scores["score"]]), "[1.0, 0.0]"
        )

    def test2(self) -> None:
        """
        Test that a reversed single edge costs 1 as a DAG, 0 as a class.
        """
        expected = "is_dag=True shd_dag=1 shd_cpdag=0 shd_target=0"
        scores = self.helper(
            2, [(0, 1)], [(1, 0)], tscore.TARGET_CPDAG, expected
        )
        legacy = [scores["f1"], scores["fdr"], scores["skeleton_f1"]]
        self.assert_equal(str(legacy), "[0.0, 1.0, 1.0]")

    def test3(self) -> None:
        """
        Test a chain against its Markov-equivalent fork: tie on the class.
        """
        expected = "is_dag=True shd_dag=1 shd_cpdag=0 shd_target=0"
        self.helper(
            3, [(0, 1), (1, 2)], [(1, 0), (1, 2)], tscore.TARGET_CPDAG, expected
        )

    def test4(self) -> None:
        """
        Test a chain against a collider: different classes.
        """
        expected = "is_dag=True shd_dag=1 shd_cpdag=2 shd_target=2"
        self.helper(
            3, [(0, 1), (1, 2)], [(0, 1), (2, 1)], tscore.TARGET_CPDAG, expected
        )

    def test5(self) -> None:
        """
        Test a fully reversed chain (the paper's Key Example 1 situation).
        """
        expected = "is_dag=True shd_dag=2 shd_cpdag=0 shd_target=0"
        self.helper(
            3, [(0, 1), (1, 2)], [(2, 1), (1, 0)], tscore.TARGET_CPDAG, expected
        )

    def test6(self) -> None:
        """
        Test a true collider against a chain through the collider.
        """
        expected = "is_dag=True shd_dag=1 shd_cpdag=2 shd_target=2"
        self.helper(
            3, [(0, 2), (1, 2)], [(0, 2), (2, 1)], tscore.TARGET_CPDAG, expected
        )

    def test7(self) -> None:
        """
        Test Meek rule R1: the edge out of a collider is compelled.
        """
        expected = "is_dag=True shd_dag=1 shd_cpdag=1 shd_target=1"
        self.helper(
            4,
            [(0, 2), (1, 2), (2, 3)],
            [(0, 2), (1, 2), (3, 2)],
            tscore.TARGET_CPDAG,
            expected,
        )

    def test8(self) -> None:
        """
        Test the DAG target: a reversal inside the class still costs 1.
        """
        expected = "is_dag=True shd_dag=1 shd_cpdag=0 shd_target=1"
        self.helper(
            3, [(0, 1), (1, 2)], [(1, 0), (1, 2)], tscore.TARGET_DAG, expected
        )

    def test9(self) -> None:
        """
        Test a cyclic output: structural scores only, no combined score.
        """
        expected = "is_dag=False shd_dag=1 shd_cpdag=3 shd_target=3"
        scores = self.helper(
            3,
            [(0, 1), (1, 2)],
            [(0, 1), (1, 2), (2, 0)],
            tscore.TARGET_CPDAG,
            expected,
        )
        self.assert_equal(str(np.isnan(scores["score"])), "True")

    def test10(self) -> None:
        """
        Test the empty graph against a 6-edge truth: score -1, pred_frac 0.
        """
        # Prepare inputs.
        true_edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5)]
        truth = _truth(_adjacency(6, true_edges), tscore.TARGET_DAG)
        empty = np.zeros((6, 6), dtype=int)
        # Run test.
        scores = tscore.score_graph(empty, truth)
        # Check outputs.
        actual = [
            scores["shd_dag"],
            scores["pred_frac"],
            round(scores["score"], 12),
        ]
        self.assert_equal(str(actual), "[6, 0.0, -1.0]")

    def test11(self) -> None:
        """
        Test an empty truth: prediction uninformative, empty graph scores 0.
        """
        # Prepare inputs.
        truth = _truth(np.zeros((3, 3), dtype=int), tscore.TARGET_CPDAG)
        empty = np.zeros((3, 3), dtype=int)
        # Run test.
        scores = tscore.score_graph(empty, truth)
        # Check outputs.
        actual = [
            scores["shd_dag"],
            scores["prediction_informative"],
            scores["score"],
        ]
        self.assert_equal(str(actual), "[0, False, 0.0]")

    def test12(self) -> None:
        """
        Test that a complete DAG along any order gets the same prediction term.
        """
        # Prepare inputs.
        true_edges = [(0, 1), (1, 2), (2, 3)]
        truth = _truth(_adjacency(4, true_edges), tscore.TARGET_CPDAG)
        complete_a = np.triu(np.ones((4, 4), dtype=int), 1)
        perm = [2, 0, 3, 1]
        complete_b = complete_a[np.ix_(perm, perm)]
        # Run test.
        ll_a = tscore.score_graph(complete_a, truth)["pred_ll"]
        ll_b = tscore.score_graph(complete_b, truth)["pred_ll"]
        # Check outputs.
        self.assert_equal(str(bool(np.isclose(ll_a, ll_b, rtol=1e-9))), "True")

    def test13(self) -> None:
        """
        Test that an exactly deterministic node is flagged, with a finite score.
        """
        # Prepare inputs.
        rng = np.random.default_rng(0)
        x = rng.normal(size=(3000, 3))
        x[:, 2] = x[:, 0] + x[:, 1]
        B = _adjacency(3, [(0, 2), (1, 2)])
        nongauss = np.zeros(3, dtype=bool)
        truth = tscore.make_truth(
            B, tscore.TARGET_CPDAG, nongauss, x[:2000], x[2000:]
        )
        # Run test.
        scores = tscore.score_graph(B, truth)
        # Check outputs.
        actual = [scores["degenerate"], bool(np.isfinite(scores["score"]))]
        self.assert_equal(str(actual), "[True, True]")


# #############################################################################
# Test_cpdag_from_dag
# #############################################################################


class Test_cpdag_from_dag(hunitest.TestCase):
    """
    Test `tukey_scoring.cpdag_from_dag()` on Meek-rule cases and at random.
    """

    def test1(self) -> None:
        """
        Test Meek rule R2: every edge of this graph is compelled.
        """
        # Prepare inputs.
        B = _adjacency(4, [(0, 2), (1, 2), (2, 3), (0, 3)])
        # Run test.
        A = tscore.cpdag_from_dag(B)
        # Check outputs.
        self.assert_equal(str(A.tolist()), str(B.tolist()))

    def test2(self) -> None:
        """
        Test a triangle: no v-structure, every edge undirected.
        """
        # Prepare inputs.
        B = _adjacency(3, [(0, 1), (1, 2), (0, 2)])
        # Run test.
        A = tscore.cpdag_from_dag(B)
        # Check outputs.
        expected = (B + B.T).tolist()
        self.assert_equal(str(A.tolist()), str(expected))

    def test3(self) -> None:
        """
        Test Meek rule R3: the edge into the double collider is compelled.
        """
        # Prepare inputs.
        B = _adjacency(4, [(0, 1), (0, 2), (0, 3), (2, 1), (3, 1)])
        # Run test.
        marks = tscore.marks_from_pdag(tscore.cpdag_from_dag(B))
        # Check outputs.
        actual = [
            int(marks[i, j]) for i, j in [(0, 1), (1, 2), (1, 3), (0, 2), (0, 3)]
        ]
        expected = [
            tscore.MARK_FORWARD,
            tscore.MARK_BACKWARD,
            tscore.MARK_BACKWARD,
            tscore.MARK_UNDIRECTED,
            tscore.MARK_UNDIRECTED,
        ]
        self.assert_equal(str(actual), str(expected))

    def test4(self) -> None:
        """
        Test against brute-force class enumeration and causal-learn.
        """
        # Prepare inputs.
        rng = np.random.default_rng(7)
        n_mismatch_bf = 0
        n_mismatch_cl = 0
        # Run test.
        for _ in range(200):
            B = _random_dag(rng, int(rng.integers(3, 6)), 0.5)
            ours = tscore.cpdag_from_dag(B)
            n_mismatch_bf += int(not np.array_equal(ours, _brute_force_cpdag(B)))
            n_mismatch_cl += int(not np.array_equal(ours, _causallearn_cpdag(B)))
        # Check outputs.
        self.assert_equal(str([n_mismatch_bf, n_mismatch_cl]), "[0, 0]")


# #############################################################################
# Test_ngdag_pattern_from_dag
# #############################################################################


class Test_ngdag_pattern_from_dag(hunitest.TestCase):
    """
    Test `tukey_scoring.ngdag_pattern_from_dag()` (Hoyer et al.

    2008).
    """

    def test1(self) -> None:
        """
        Test a chain whose last node is non-Gaussian.
        """
        # Prepare inputs.
        B = _adjacency(3, [(0, 1), (1, 2)])
        nongauss = np.array([False, False, True])
        # Run test.
        marks = tscore.marks_from_pdag(
            tscore.ngdag_pattern_from_dag(B, nongauss)
        )
        # Check outputs.
        actual = [int(marks[0, 1]), int(marks[1, 2])]
        expected = [tscore.MARK_UNDIRECTED, tscore.MARK_FORWARD]
        self.assert_equal(str(actual), str(expected))

    def test2(self) -> None:
        """
        Test a chain whose first node is non-Gaussian: fully oriented.
        """
        # Prepare inputs.
        B = _adjacency(3, [(0, 1), (1, 2)])
        nongauss = np.array([True, False, False])
        # Run test.
        marks = tscore.marks_from_pdag(
            tscore.ngdag_pattern_from_dag(B, nongauss)
        )
        # Check outputs.
        actual = [int(marks[0, 1]), int(marks[1, 2])]
        expected = [tscore.MARK_FORWARD, tscore.MARK_FORWARD]
        self.assert_equal(str(actual), str(expected))


# #############################################################################
# Test_score_pdag
# #############################################################################


class Test_score_pdag(hunitest.TestCase):
    """
    Test `tukey_scoring.score_pdag()` on undirected outputs.
    """

    def test1(self) -> None:
        """
        Test an undirected edge: matches a class target, not a DAG target.
        """
        # Prepare inputs.
        B = _adjacency(2, [(0, 1)])
        A_hat = np.array([[0, 1], [1, 0]])
        truth_class = _truth(B, tscore.TARGET_CPDAG)
        truth_dag = _truth(B, tscore.TARGET_DAG)
        # Run test.
        scores_class = tscore.score_pdag(A_hat, truth_class)
        scores_dag = tscore.score_pdag(A_hat, truth_dag)
        # Check outputs.
        actual = [
            scores_class["shd_target"],
            scores_dag["shd_target"],
            scores_class["extendable"],
        ]
        self.assert_equal(str(actual), "[0, 1, True]")


# #############################################################################
# Test_score_properties
# #############################################################################


class Test_score_properties(hunitest.TestCase):
    """
    Property tests: prototype SHD parity, class ties, relabelling, ordering.
    """

    def test1(self) -> None:
        """
        Test mark SHD equals the prototype's `shd()` and causal-learn's `SHD`.
        """
        # Prepare inputs.
        rng = np.random.default_rng(11)
        n_mismatch_proto = 0
        n_mismatch_cl = 0
        # Run test.
        for idx in range(1000):
            d = int(rng.integers(2, 7))
            B1 = _random_dag(rng, d, 0.4)
            B2 = _random_dag(rng, d, 0.4)
            ours = tscore.shd_marks(
                tscore.marks_from_pdag(B1), tscore.marks_from_pdag(B2)
            )
            n_mismatch_proto += int(ours != rlproto.shd(B2, B1))
            if idx < 100:
                nodes = [cgrgnod.GraphNode(f"x{i}") for i in range(d)]
                g1, g2 = cgrdag.Dag(nodes), cgrdag.Dag(nodes)
                for i, j in np.argwhere(B1):
                    g1.add_directed_edge(nodes[i], nodes[j])
                for i, j in np.argwhere(B2):
                    g2.add_directed_edge(nodes[i], nodes[j])
                n_mismatch_cl += int(ours != cgrshd.SHD(g1, g2).get_shd())
        # Check outputs.
        self.assert_equal(str([n_mismatch_proto, n_mismatch_cl]), "[0, 0]")

    def test2(self) -> None:
        """
        Test every member of an equivalence class gets the same prediction term.
        """
        # Prepare inputs.
        rng = np.random.default_rng(3)
        max_spread = 0.0
        # Run test.
        for noise in ("gaussian", "exponential"):
            for seed in range(5):
                B = _random_dag(rng, 5, 0.5)
                truth = _truth(B, tscore.TARGET_CPDAG, seed=seed, noise=noise)
                A = tscore.cpdag_from_dag(B)
                values = []
                for flips in itertools.product(
                    [False, True], repeat=int(np.triu(A & A.T, 1).sum())
                ):
                    member = ((A == 1) & (A.T == 0)).astype(int)
                    for (i, j), flip in zip(
                        np.argwhere(np.triu(A & A.T, 1)), flips
                    ):
                        member[j, i] = 1 if flip else 0
                        member[i, j] = 0 if flip else 1
                    if tscore.is_dag(member) and _v_structures(
                        member
                    ) == _v_structures(B):
                        values.append(
                            tscore.score_graph(member, truth)["pred_ll"]
                        )
                spread = (max(values) - min(values)) / max(1.0, abs(values[0]))
                max_spread = max(max_spread, spread)
        # Check outputs.
        self.assert_equal(str(bool(max_spread < 1e-8)), "True")

    def test3(self) -> None:
        """
        Test relabelling the nodes changes no score.
        """
        # Prepare inputs.
        rng = np.random.default_rng(5)
        B = _random_dag(rng, 5, 0.5)
        B_hat = _random_dag(rng, 5, 0.5)
        x_train, x_test = _simulate(B, seed=1)
        nongauss = np.zeros(5, dtype=bool)
        perm = rng.permutation(5)
        truth = tscore.make_truth(
            B, tscore.TARGET_CPDAG, nongauss, x_train, x_test
        )
        truth_perm = tscore.make_truth(
            B[np.ix_(perm, perm)],
            tscore.TARGET_CPDAG,
            nongauss,
            x_train[:, perm],
            x_test[:, perm],
        )
        # Run test.
        scores = tscore.score_graph(B_hat, truth, legacy=False)
        scores_perm = tscore.score_graph(
            B_hat[np.ix_(perm, perm)], truth_perm, legacy=False
        )
        # Check outputs.
        same = all(
            np.isclose(scores[k], scores_perm[k])
            for k in ["shd_dag", "shd_cpdag", "pred_ll", "score"]
        )
        self.assert_equal(str(same), "True")

    def test4(self) -> None:
        """
        Test the ordering G* >= any orientation of its skeleton, ties in class.
        """
        # Prepare inputs.
        rng = np.random.default_rng(9)
        n_violations = 0
        n_ties_outside_class = 0
        # Run test.
        for seed in range(30):
            B = _random_dag(rng, 5, 0.5)
            if B.sum() == 0:
                continue
            target = [tscore.TARGET_DAG, tscore.TARGET_CPDAG][seed % 2]
            truth = _truth(B, target, seed=seed)
            best = tscore.score_graph(B, truth, legacy=False)["score"]
            marks_true = tscore.target_marks(B, target, truth.nongauss)
            for _ in range(10):
                orient = np.triu((B + B.T) > 0, 1).astype(int)
                flip = np.triu(rng.random(B.shape) < 0.5, 1) & (orient == 1)
                cand = orient - flip + flip.T
                if not tscore.is_dag(cand):
                    continue
                score = tscore.score_graph(cand, truth, legacy=False)["score"]
                same_class = np.array_equal(
                    tscore.target_marks(cand, target, truth.nongauss), marks_true
                )
                n_violations += int(score > best + 1e-9)
                n_ties_outside_class += int(
                    not same_class and abs(score - best) < 1e-9
                )
        # Check outputs.
        self.assert_equal(str([n_violations, n_ties_outside_class]), "[0, 0]")
