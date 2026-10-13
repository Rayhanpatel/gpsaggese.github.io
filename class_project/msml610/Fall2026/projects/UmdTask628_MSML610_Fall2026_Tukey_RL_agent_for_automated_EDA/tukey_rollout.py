"""
Play many episodes in lockstep so the policy evaluates states in batches.

Episodes are grouped by number of variables; within a group every step runs
one batched forward pass of the tool tower. The same runner serves behaviour
cloning data (scripted expert), decoder states (policy with random
exploration), tool-policy rollouts and greedy evaluation.

Import as:

import tukey_rollout as troll
"""

import collections
import dataclasses
import logging
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

import helpers.hdbg as hdbg
import tukey_env as tenv
import tukey_features as tfeat
import tukey_policy as tpolic
import tukey_scripted as tscrip
import tukey_tools as ttools

_LOG = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class Task:
    """
    One episode to play: generator settings, instance seed and tool budget.
    """

    cfg: tenv.GraphConfig
    seed: int
    budget: int


@dataclasses.dataclass
class Decision:
    """
    One decision: the state, its legal actions and the chosen candidate.
    """

    obs: tfeat.Observation
    legal: np.ndarray
    # Index in `TukeyPolicy.candidates(d)`.
    action_index: int
    # Value-head prediction at the state (0 for the scripted expert).
    value: float


@dataclasses.dataclass
class Episode:
    """
    A finished episode.
    """

    task: Task
    decisions: List[Decision]
    # Environment at the end of the episode (`submit_graph()` scores a graph).
    env: tenv.TukeyEnv
    final_evidence: ttools.Evidence
    final_obs: tfeat.Observation
    # Exact cost-free score of any DAG on this instance (trainer-only).
    score_fn: Callable[[np.ndarray], float]

    @property
    def n_calls(self) -> int:
        """
        Number of tool calls made.
        """
        return sum(d.action_index != len(d.legal) - 1 for d in self.decisions)

    @property
    def calls_from(self) -> List[int]:
        """
        Tool calls made at or after each decision.
        """
        is_call = [
            int(d.action_index != len(d.legal) - 1) for d in self.decisions
        ]
        calls_from = np.cumsum(is_call[::-1])[::-1] if is_call else []
        return [int(c) for c in calls_from]


def group_by_nodes(items: List, n_nodes: Callable) -> Dict[int, List[int]]:
    """
    Group item positions by number of variables.

    :param items: items
    :param n_nodes: function giving an item's number of variables
    :return: positions per number of variables
    """
    groups: Dict[int, List[int]] = collections.defaultdict(list)
    for k, item in enumerate(items):
        groups[n_nodes(item)].append(k)
    return dict(groups)


def _choose(
    log_probs: torch.Tensor,
    legal: np.ndarray,
    *,
    greedy: bool,
    eps_random: float,
    generator: torch.Generator,
) -> np.ndarray:
    """
    Pick one candidate per state.

    :param log_probs: log-probabilities (B, n_candidates)
    :param legal: legal masks (B, n_candidates)
    :param greedy: take the most likely candidate
    :param eps_random: probability mass moved to uniform legal choices
    :param generator: random generator
    :return: chosen candidate indices (B,)
    """
    if greedy:
        return torch.argmax(log_probs, dim=-1).numpy()
    probs = log_probs.exp()
    if eps_random > 0:
        legal_t = torch.as_tensor(legal, dtype=probs.dtype)
        uniform = legal_t / legal_t.sum(dim=-1, keepdim=True)
        probs = (1 - eps_random) * probs + eps_random * uniform
    choice = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
    return choice.numpy()


def _run_group(
    tasks: List[Task],
    choose_fn: Callable[
        [List[ttools.Evidence], List[tfeat.Observation], np.ndarray],
        Tuple[np.ndarray, np.ndarray],
    ],
    actions: List[ttools.Action],
) -> List[Episode]:
    """
    Play tasks with the same number of variables in lockstep.

    :param tasks: tasks
    :param choose_fn: (evidences, observations, legal) -> (choices, values)
    :param actions: candidate actions for this number of variables
    :return: finished episodes, in task order
    """
    envs = [tenv.TukeyEnv(task.cfg, budget=task.budget) for task in tasks]
    evidences = [env.reset(seed=task.seed)[0] for env, task in zip(envs, tasks)]
    score_fns = [env.reward_oracle() for env in envs]
    decisions: List[List[Decision]] = [[] for _ in tasks]
    active = [k for k, ev in enumerate(evidences) if ev.budget_left > 0]
    stop_index = len(actions) - 1
    hdbg.dassert_eq(actions[stop_index].kind, ttools.STOP)
    while active:
        obs = [tfeat.featurize(evidences[k]) for k in active]
        legal = np.stack(
            [tpolic.legal_mask(actions, evidences[k]) for k in active]
        )
        choices, values = choose_fn([evidences[k] for k in active], obs, legal)
        still_active = []
        for row, k in enumerate(active):
            choice = int(choices[row])
            hdbg.dassert(legal[row, choice], "Illegal action chosen")
            decisions[k].append(
                Decision(obs[row], legal[row], choice, float(values[row]))
            )
            if choice == stop_index:
                continue
            evidences[k], _, _, _, _ = envs[k].step(actions[choice])
            if evidences[k].budget_left > 0:
                still_active.append(k)
        active = still_active
    episodes = [
        Episode(
            task=task,
            decisions=decisions[k],
            env=envs[k],
            final_evidence=evidences[k],
            final_obs=tfeat.featurize(evidences[k]),
            score_fn=score_fns[k],
        )
        for k, task in enumerate(tasks)
    ]
    return episodes


def run_policy_episodes(
    policy: tpolic.TukeyPolicy,
    tasks: List[Task],
    *,
    greedy: bool,
    generator: torch.Generator,
    eps_random: float = 0.0,
) -> List[Episode]:
    """
    Play tasks with the policy's tool tower (no gradients).

    :param policy: policy
    :param tasks: tasks (any mix of sizes)
    :param greedy: take the most likely action
    :param generator: random generator
    :param eps_random: exploration mass on uniform legal actions
    :return: episodes in task order
    """
    episodes: List[Optional[Episode]] = [None] * len(tasks)
    groups = group_by_nodes(tasks, lambda task: task.cfg.n_nodes)
    for n_nodes, positions in groups.items():
        actions, _ = policy.candidates(n_nodes)

        def _choose_fn(evidences, obs, legal):
            batch = tpolic.stack_observations(obs, dtype=policy.dtype)
            with torch.no_grad():
                log_probs, values = policy.tool_log_probs_batch(batch, legal)
            choices = _choose(
                log_probs,
                legal,
                greedy=greedy,
                eps_random=eps_random,
                generator=generator,
            )
            return choices, values.numpy()

        group = _run_group([tasks[k] for k in positions], _choose_fn, actions)
        for k, episode in zip(positions, group):
            episodes[k] = episode
    return episodes  # type: ignore[return-value]


def run_expert_episodes(
    expert: tscrip.ScriptedExpert,
    policy: tpolic.TukeyPolicy,
    tasks: List[Task],
) -> List[Episode]:
    """
    Play tasks with the scripted expert, indexing actions like the policy.

    :param expert: scripted expert
    :param policy: policy whose candidate lists index the actions
    :param tasks: tasks (any mix of sizes)
    :return: episodes in task order
    """
    episodes: List[Optional[Episode]] = [None] * len(tasks)
    groups = group_by_nodes(tasks, lambda task: task.cfg.n_nodes)
    for n_nodes, positions in groups.items():
        actions, _ = policy.candidates(n_nodes)
        position = {action: k for k, action in enumerate(actions)}

        def _choose_fn(evidences, obs, legal):
            choices = [position[expert.next_action(ev)] for ev in evidences]
            return np.array(choices), np.zeros(len(choices))

        group = _run_group([tasks[k] for k in positions], _choose_fn, actions)
        for k, episode in zip(positions, group):
            episodes[k] = episode
    return episodes  # type: ignore[return-value]


# #############################################################################
# Evaluation
# #############################################################################


def evaluate_policy(
    policy: tpolic.TukeyPolicy,
    tasks: List[Task],
    *,
    generator: torch.Generator,
    decoder_policy: Optional[tpolic.TukeyPolicy] = None,
) -> List[Dict]:
    """
    Play tasks greedily, decode the greedy graph and score it.

    :param policy: policy whose tool tower plays
    :param tasks: tasks
    :param generator: random generator (unused by greedy play)
    :param decoder_policy: policy whose decoder emits the graph
        - Default: None means `policy`
    :return: score dictionary per task (`TukeyEnv.submit_graph()`)
    """
    decoder_policy = decoder_policy or policy
    episodes = run_policy_episodes(
        policy, tasks, greedy=True, generator=generator
    )
    results: List[Optional[Dict]] = [None] * len(tasks)
    groups = group_by_nodes(episodes, lambda e: e.task.cfg.n_nodes)
    for positions in groups.values():
        batch = tpolic.stack_observations(
            [episodes[k].final_obs for k in positions],
            dtype=decoder_policy.dtype,
        )
        with torch.no_grad():
            u, logits = decoder_policy.decoder_params_batch(batch)
        for row, k in enumerate(positions):
            B = tpolic.greedy_graph(u[row], logits[row])
            edges = [(int(i), int(j)) for i, j in np.argwhere(B == 1)]
            results[k] = episodes[k].env.submit_graph(edges)
    return results  # type: ignore[return-value]


def evaluate_expert(
    expert: tscrip.ScriptedExpert,
    policy: tpolic.TukeyPolicy,
    tasks: List[Task],
) -> List[Dict]:
    """
    Play tasks with the scripted expert and score its projected DAG.

    :param expert: scripted expert
    :param policy: policy whose candidate lists index the actions
    :param tasks: tasks
    :return: score dictionary per task
    """
    episodes = run_expert_episodes(expert, policy, tasks)
    results = []
    for episode in episodes:
        edges = expert.emit_graph(episode.final_evidence)
        results.append(episode.env.submit_graph(edges))
    return results
