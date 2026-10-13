"""
Gate G8 (environment and policy part): nothing hidden reaches an agent.

- Rescaling columns leaves the policy's input unchanged unless the agent calls
  `variance` (the only scale channel, kept for the variance-shortcut study)
- Test rows set to NaN change no observation, action or emitted graph
- (In `test_tukey_env.py`: `info` is whitelisted and the evidence holds only
  the training rows)

Data dependence of a trained policy (shuffled or foreign evidence must change
the emitted graph) is measured on checkpoints by the evaluation code.

Import as:

import test.test_tukey_leaks as ttetulea
"""

import dataclasses
import logging

import numpy as np
import torch

import helpers.hunit_test as hunitest
import tukey_env as tenv
import tukey_features as tfeat
import tukey_policy as tpolic
import tukey_scoring as tscore
import tukey_tools as ttools

_LOG = logging.getLogger(__name__)


def _play_greedy(policy: tpolic.TukeyPolicy, env: tenv.TukeyEnv) -> tuple:
    """
    Play a reset environment greedily and decode the greedy graph.

    :param policy: policy
    :param env: environment after `reset()`
    :return: actions taken and emitted adjacency
    """
    agent = tpolic.PolicyAgent(policy, greedy=True)
    evidence = env._evidence
    actions = []
    while evidence.budget_left > 0:
        action = agent.next_action(evidence)
        if action.kind == ttools.STOP:
            break
        actions.append(action)
        evidence, _, _, _, _ = env.step(action)
    edges = agent.emit_graph(evidence)
    B, _ = tscore.edges_to_adjacency(edges, env.cfg.n_nodes)
    return actions, B


class Test_leaks(hunitest.TestCase):
    """
    Check what can reach a policy.
    """

    def test1(self) -> None:
        """
        Rescaled columns give the same features when `variance` is not called.
        """
        cfg = tenv.GraphConfig(family="linear_exp")
        x = tenv.sample_instance(cfg, 31).d_train.to_numpy()
        rng = np.random.default_rng(31)
        scales = np.exp(rng.uniform(np.log(0.1), np.log(10.0), size=6))
        actions = [
            a
            for a in ttools.enumerate_tool_actions(6)
            if a.kind != ttools.VARIANCE
        ]
        picked = [
            actions[k] for k in rng.choice(len(actions), 50, replace=False)
        ]
        features = []
        for data in (x, x * scales + 3.0):
            evidence = ttools.Evidence(ttools.ToolCache(data), budget=60)
            for action in picked:
                evidence.record(action)
            features.append(tfeat.featurize(evidence))
        for name in ("node", "pair", "glob"):
            np.testing.assert_allclose(
                getattr(features[0], name), getattr(features[1], name), atol=1e-6
            )

    def test2(self) -> None:
        """
        NaN test rows change no action and no emitted graph.
        """
        torch.manual_seed(0)
        policy = tpolic.TukeyPolicy(hidden=16)
        cfg = tenv.GraphConfig(family="mixed")
        env = tenv.TukeyEnv(cfg, budget=8)
        env.reset(seed=32)
        actions_1, B_1 = _play_greedy(policy, env)
        env = tenv.TukeyEnv(cfg, budget=8)
        env.reset(seed=32)
        inst = env._instance
        nan_test = inst.d_test * np.nan
        truth = dataclasses.replace(
            inst.truth,
            x_test=inst.truth.x_test * np.nan,
            moments_test=inst.truth.moments_test * np.nan,
        )
        env._instance = dataclasses.replace(inst, d_test=nan_test, truth=truth)
        actions_2, B_2 = _play_greedy(policy, env)
        self.assertEqual(actions_1, actions_2)
        self.assertTrue(np.array_equal(B_1, B_2))
