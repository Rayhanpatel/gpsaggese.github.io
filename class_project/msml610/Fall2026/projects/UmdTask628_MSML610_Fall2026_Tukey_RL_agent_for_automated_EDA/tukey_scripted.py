"""
Non-learned agents: the prototype's scripted rule and its tool-API version.

- `proto_faithful_edges()` runs the prototype's own `simulate_agent_output()`
  on the training rows; it is the reference implementation of the rule
- `ScriptedExpert` is the same rule as a state -> action function over
  `Evidence`: it plays through `TukeyEnv` with real tool calls, can run under a
  budget, and is the behaviour-cloning teacher
- `project_to_dag()` removes cycles with a fixed rule (weakest `|r|` first)
- `run_episode()` plays an agent in an environment and scores its graph

The rule (prototype, `simulate_agent_output()`): for each pair i < j in index
order, keep the pair if `|r| >= 0.15` and no single other variable k gives
`|pcorr(i, j | k)| < 0.15` (tested in index order, stopping at the first that
removes it); orient a kept pair as i -> j if the residual of j on i depends
less on i than the residual of i on j depends on j.

Import as:

import tukey_scripted as tscrip
"""

import itertools
import json
import logging
import os
import tempfile
import types
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import helpers.hdbg as hdbg
import research.agentic_data_science.RL_for_auto_EDA_prototype as rlproto
import tukey_env as tenv
import tukey_scoring as tscore
import tukey_tools as ttools

_LOG = logging.getLogger(__name__)

# Threshold of the prototype for |r| and |pcorr|.
PROTO_THRESHOLD = 0.15


# #############################################################################
# Reference implementation
# #############################################################################


def proto_faithful_edges(x_train: np.ndarray) -> List[Tuple[int, int]]:
    """
    Run the prototype's scripted agent on training rows.

    :param x_train: training rows, shape (n_train, n_nodes)
    :return: edges `[(parent, child), ...]` exactly as the prototype emits them
    """
    env_like = types.SimpleNamespace(X_train=x_train, n_nodes=x_train.shape[1])
    with tempfile.TemporaryDirectory() as workdir:
        rlproto.simulate_agent_output(env_like, workdir)
        with open(os.path.join(workdir, "graph.json")) as file:
            edges = json.load(file)
    edges = [(int(parent), int(child)) for parent, child in edges]
    return edges


# #############################################################################
# Projection
# #############################################################################


def _find_cycle(B: np.ndarray) -> Optional[List[int]]:
    """
    Find one directed cycle.

    :param B: directed adjacency
    :return: nodes of a cycle in order, or None if `B` is acyclic
    """
    n_nodes = B.shape[0]
    # 0 = unvisited, 1 = on the current path, 2 = done.
    state = [0] * n_nodes
    path: List[int] = []

    def _visit(node: int) -> Optional[List[int]]:
        state[node] = 1
        path.append(node)
        for child in np.flatnonzero(B[node]):
            child = int(child)
            if state[child] == 1:
                return path[path.index(child) :]
            if state[child] == 0:
                cycle = _visit(child)
                if cycle is not None:
                    return cycle
        state[node] = 2
        path.pop()
        return None

    for start in range(n_nodes):
        if state[start] == 0:
            cycle = _visit(start)
            if cycle is not None:
                return cycle
    return None


def project_to_dag(
    edges: List[Tuple[int, int]], corr: np.ndarray
) -> Tuple[List[Tuple[int, int]], int]:
    """
    Remove cycles by deleting the weakest edge on each cycle found.

    The weakest edge has the smallest `|r|`; ties go to the smallest
    `(parent, child)`. Used only for agents whose rule can emit cycles.

    :param edges: `[(parent, child), ...]`
    :param corr: correlation matrix of the training rows
    :return: acyclic edge list and the number of deleted edges
    """
    n_nodes = corr.shape[0]
    B, _ = tscore.edges_to_adjacency(edges, n_nodes)
    n_deleted = 0
    cycle = _find_cycle(B)
    while cycle is not None:
        cycle_edges = [
            (cycle[k], cycle[(k + 1) % len(cycle)]) for k in range(len(cycle))
        ]
        weakest = min(cycle_edges, key=lambda e: (abs(corr[e[0], e[1]]), e))
        B[weakest] = 0
        n_deleted += 1
        cycle = _find_cycle(B)
    projected = [(int(i), int(j)) for i, j in np.argwhere(B == 1)]
    return projected, n_deleted


# #############################################################################
# ScriptedExpert
# #############################################################################


class ScriptedExpert:
    """
    The prototype's rule as a state -> action function over `Evidence`.

    `next_action()` returns the first call of the rule whose result is not yet
    in the evidence, so the agent can be interrupted by a budget at any point;
    `emit_graph()` turns whatever evidence exists into a DAG.
    """

    def __init__(self, *, threshold: float = PROTO_THRESHOLD) -> None:
        """
        Build the agent.

        :param threshold: cut-off for `|r|` and `|pcorr|`
            - Default: the prototype's 0.15
        """
        self.threshold = threshold

    def _ci_actions(
        self, evidence: ttools.Evidence, i: int, j: int
    ) -> List[ttools.Action]:
        """
        List the pair's CI tests in the rule's order.

        :param evidence: evidence so far
        :param i: first variable
        :param j: second variable
        :return: `ci_test` actions with each single other variable
        """
        actions = [
            ttools.ci_test_action(i, j, (k,))
            for k in range(evidence.n_nodes)
            if k not in (i, j)
        ]
        return actions

    def _next_pair_action(
        self, evidence: ttools.Evidence, i: int, j: int
    ) -> Optional[ttools.Action]:
        """
        Return the pair's next missing call, or None if the pair is finished.

        :param evidence: evidence so far
        :param i: first variable
        :param j: second variable
        :return: next action for the pair, or None
        """
        for action in self._ci_actions(evidence, i, j):
            result = evidence.results.get(action)
            if result is None:
                return action
            if abs(result["pcorr"]) < self.threshold:
                # Removed: no orientation needed.
                return None
        for action in (
            ttools.regress_action(j, (i,)),
            ttools.regress_action(i, (j,)),
        ):
            if action not in evidence.results:
                return action
        return None

    def next_action(self, evidence: ttools.Evidence) -> ttools.Action:
        """
        Return the rule's next tool call, or STOP when done or out of budget.

        :param evidence: evidence so far
        :return: action
        """
        stop = ttools.Action(ttools.STOP)
        if evidence.budget_left <= 0:
            return stop
        for i, j in itertools.combinations(range(evidence.n_nodes), 2):
            if abs(evidence.corr[i, j]) < self.threshold:
                continue
            action = self._next_pair_action(evidence, i, j)
            if action is not None:
                return action
        return stop

    def _is_removed(self, evidence: ttools.Evidence, i: int, j: int) -> bool:
        """
        Check whether an observed CI test removed the pair.

        :param evidence: evidence so far
        :param i: first variable
        :param j: second variable
        :return: True if some observed `|pcorr|` is below the threshold
        """
        for action in self._ci_actions(evidence, i, j):
            result = evidence.results.get(action)
            if result is not None and abs(result["pcorr"]) < self.threshold:
                return True
        return False

    def _is_kept(self, evidence: ttools.Evidence, i: int, j: int) -> bool:
        """
        Check whether the pair is in the emitted skeleton.

        :param evidence: evidence so far
        :param i: first variable
        :param j: second variable
        :return: True if the pair passes the screen and was not removed
        """
        passes_screen = abs(evidence.corr[i, j]) >= self.threshold
        kept = passes_screen and not self._is_removed(evidence, i, j)
        return kept

    def _orient(
        self, evidence: ttools.Evidence, i: int, j: int
    ) -> Tuple[int, int]:
        """
        Orient a kept pair from the evidence.

        :param evidence: evidence so far
        :param i: lower-index variable
        :param j: higher-index variable
        :return: `(parent, child)`
        """
        dep_ij = evidence.results.get(ttools.regress_action(j, (i,)))
        dep_ji = evidence.results.get(ttools.regress_action(i, (j,)))
        lingam = evidence.results.get(ttools.Action(ttools.LINGAM_DIR, (i, j)))
        if dep_ij is not None and dep_ji is not None:
            # As in the prototype: i -> j if dep(i, j) < dep(j, i).
            forward = dep_ij["resid_dep"] < dep_ji["resid_dep"]
        elif lingam is not None:
            forward = lingam["stat"] > 0
        else:
            forward = True
        edge = (i, j) if forward else (j, i)
        return edge

    def emit_raw_edges(self, evidence: ttools.Evidence) -> List[Tuple[int, int]]:
        """
        Emit the rule's edges from the evidence, possibly with cycles.

        A pair is kept unless an observed CI test removed it (untested pairs
        that pass the free `|r|` screen are kept); a kept pair is oriented by
        its two regressions if both were run, else by `lingam_dir` if run,
        else from the lower to the higher column index (columns are randomly
        permuted, so this is an uninformed guess).

        :param evidence: evidence so far
        :return: edges `[(parent, child), ...]` in pair order
        """
        edges = []
        for i, j in itertools.combinations(range(evidence.n_nodes), 2):
            if self._is_kept(evidence, i, j):
                edges.append(self._orient(evidence, i, j))
        return edges

    def emit_graph(self, evidence: ttools.Evidence) -> List[Tuple[int, int]]:
        """
        Emit a DAG: the rule's edges with cycles removed by `project_to_dag()`.

        :param evidence: evidence so far
        :return: acyclic edges `[(parent, child), ...]`
        """
        edges, _ = project_to_dag(self.emit_raw_edges(evidence), evidence.corr)
        return edges


# #############################################################################
# Episodes
# #############################################################################


def run_episode(
    env: tenv.TukeyEnv, agent: Any, *, seed: int
) -> Tuple[Dict[str, Any], List[ttools.Action]]:
    """
    Play an agent for one episode and score its graph.

    :param env: environment
    :param agent: object with `next_action(evidence)` and `emit_graph(evidence)`
    :param seed: instance seed
    :return: score dictionary and the tool calls made
    """
    evidence, _ = env.reset(seed=seed)
    while True:
        action = agent.next_action(evidence)
        evidence, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    edges = agent.emit_graph(evidence)
    B_hat, _ = tscore.edges_to_adjacency(edges, env.cfg.n_nodes)
    hdbg.dassert(tscore.is_dag(B_hat), "Agents must emit a DAG")
    scores = env.submit_graph(edges)
    return scores, list(evidence.history)
