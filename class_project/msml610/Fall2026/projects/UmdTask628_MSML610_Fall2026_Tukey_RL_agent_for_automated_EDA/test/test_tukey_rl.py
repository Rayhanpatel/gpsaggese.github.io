"""
Test the batched rollouts and training steps against the single-state losses.

The single-state losses are the ones the gradient checks verify; the batched
versions used for training must give the same values and gradients.

Import as:

import test.test_tukey_rl as ttetutrl
"""

import logging
from typing import List

import numpy as np
import torch

import helpers.hunit_test as hunitest
import tukey_env as tenv
import tukey_policy as tpolic
import tukey_rl as trl
import tukey_rollout as troll
import tukey_scripted as tscrip
import tukey_tools as ttools

_LOG = logging.getLogger(__name__)


def _tasks(n_tasks: int, *, budget: int = 12) -> List[troll.Task]:
    """
    Build tasks of mixed sizes.

    :param n_tasks: number of tasks
    :param budget: tool budget
    :return: tasks
    """
    tasks = []
    for k in range(n_tasks):
        n_nodes = (4, 5)[k % 2]
        cfg = tenv.GraphConfig(family="mixed", n_nodes=n_nodes, n_edges=n_nodes)
        tasks.append(troll.Task(cfg=cfg, seed=100 + k, budget=budget))
    return tasks


def _expert_dag(
    expert: tscrip.ScriptedExpert, episode: troll.Episode
) -> np.ndarray:
    """
    Return the expert's emitted DAG at the end of an episode.

    :param expert: scripted expert
    :param episode: expert episode
    :return: DAG adjacency
    """
    n_nodes = episode.task.cfg.n_nodes
    B = np.zeros((n_nodes, n_nodes), dtype=int)
    for parent, child in expert.emit_graph(episode.final_evidence):
        B[parent, child] = 1
    return B


def _grads(policy: tpolic.TukeyPolicy) -> torch.Tensor:
    """
    Flatten the current gradients of the tool tower.

    :param policy: policy
    :return: concatenated gradients
    """
    params = list(policy.tool_encoder.parameters())
    params += list(policy.tool_head.parameters())
    return torch.cat([p.grad.flatten() for p in params])


class Test_rollouts(hunitest.TestCase):
    """
    Check the lockstep episode runner.
    """

    def test1(self) -> None:
        """
        Expert episodes in lockstep equal sequential expert play.
        """
        torch.manual_seed(0)
        policy = tpolic.TukeyPolicy(hidden=16)
        expert = tscrip.ScriptedExpert()
        tasks = _tasks(4, budget=15)
        episodes = troll.run_expert_episodes(expert, policy, tasks)
        for task, episode in zip(tasks, episodes):
            env = tenv.TukeyEnv(task.cfg, budget=task.budget)
            _, history = tscrip.run_episode(env, expert, seed=task.seed)
            actions, _ = policy.candidates(task.cfg.n_nodes)
            played = [actions[d.action_index] for d in episode.decisions]
            tool_calls = [a for a in played if a.kind != ttools.STOP]
            self.assertEqual(tool_calls, history)

    def test2(self) -> None:
        """
        Policy episodes respect budgets; greedy play is deterministic.
        """
        torch.manual_seed(1)
        policy = tpolic.TukeyPolicy(hidden=16)
        tasks = _tasks(6, budget=5)
        generator = torch.Generator().manual_seed(0)
        greedy1 = troll.run_policy_episodes(
            policy, tasks, greedy=True, generator=generator
        )
        greedy2 = troll.run_policy_episodes(
            policy, tasks, greedy=True, generator=generator
        )
        for episode1, episode2 in zip(greedy1, greedy2):
            self.assertLessEqual(episode1.n_calls, 5)
            self.assertEqual(
                [d.action_index for d in episode1.decisions],
                [d.action_index for d in episode2.decisions],
            )


class Test_batched_losses(hunitest.TestCase):
    """
    Batched training steps equal the single-state losses.
    """

    def test1(self) -> None:
        """
        Batched behaviour cloning equals the per-state losses.
        """
        torch.manual_seed(2)
        policy = tpolic.TukeyPolicy(hidden=16).double()
        expert = tscrip.ScriptedExpert()
        episodes = troll.run_expert_episodes(expert, policy, _tasks(4))
        decisions = [d for e in episodes for d in e.decisions]
        loss_batch, _ = trl.accumulate_bc_tool(policy, decisions)
        steps = [
            trl.ExpertStep(d.obs, d.legal, d.action_index) for d in decisions
        ]
        loss_single = float(trl.bc_tool_loss(policy, steps))
        self.assertAlmostEqual(loss_batch, loss_single, places=10)
        finals = [(e.final_obs, _expert_dag(expert, e)) for e in episodes]
        dec_batch = trl.accumulate_bc_decoder(policy, finals)
        dec_single = np.mean(
            [float(trl.bc_decoder_loss(policy, obs, B)) for obs, B in finals]
        )
        self.assertAlmostEqual(dec_batch, dec_single, places=10)

    def test2(self) -> None:
        """
        The batched tool gradient equals `tool_reinforce_loss`'s gradient.
        """
        for shaping in (False, True):
            torch.manual_seed(3)
            policy = tpolic.TukeyPolicy(hidden=16).double()
            generator = torch.Generator().manual_seed(4)
            tasks = _tasks(2, budget=6)
            groups = [
                troll.run_policy_episodes(
                    policy, [task] * 3, greedy=False, generator=generator
                )
                for task in tasks
            ]
            policy.zero_grad()
            trl.accumulate_tool_rl(
                policy,
                policy,
                groups,
                cost=0.01,
                shaping=shaping,
                n_phi_samples=4,
                generator=torch.Generator().manual_seed(5),
                entropy_coef=0.0,
                value_coef=0.0,
            )
            grad_batch = _grads(policy).clone()
            # Reference: per-state log-probabilities and `tool_reinforce_loss`.
            policy.zero_grad()
            episodes = [e for group in groups for e in group]
            phi_final, first = trl.terminal_phi(
                policy,
                episodes,
                n_samples=4,
                generator=torch.Generator().manual_seed(5),
            )
            losses = []
            position = 0
            for group in groups:
                rollouts = []
                for episode in group:
                    log_probs = [
                        policy.tool_log_probs(d.obs, d.legal)[d.action_index]
                        for d in episode.decisions
                    ]
                    rollouts.append(
                        trl.Rollout(
                            log_probs=log_probs,
                            entropies=[],
                            phi=[d.value for d in episode.decisions],
                            phi_final=float(phi_final[position]),
                            sampled_reward=float(first[position]),
                            calls_from=episode.calls_from,
                            n_calls=episode.n_calls,
                        )
                    )
                    position += 1
                losses.append(
                    trl.tool_reinforce_loss(rollouts, cost=0.01, shaping=shaping)
                )
            (sum(losses) / len(losses)).backward()
            grad_single = _grads(policy)
            np.testing.assert_allclose(
                grad_batch.numpy(), grad_single.numpy(), atol=1e-10
            )
