"""
Test the feature encoder: equivariance, masks and directional antisymmetry.

Import as:

import test.test_tukey_features as ttetufea
"""

import logging
from typing import List

import numpy as np

import helpers.hunit_test as hunitest
import tukey_env as tenv
import tukey_features as tfeat
import tukey_tools as ttools

_LOG = logging.getLogger(__name__)


def _relabel(action: ttools.Action, new_index: List[int]) -> ttools.Action:
    """
    Rewrite an action for relabelled variables.

    :param action: action on the original labels
    :param new_index: `new_index[v]` is the new label of variable v
    :return: the same call on the new labels, canonicalized
    """
    args = [new_index[v] for v in action.args]
    if action.kind == ttools.CI_TEST:
        relabelled = ttools.ci_test_action(args[0], args[1], tuple(args[2:]))
    elif action.kind == ttools.REGRESS:
        relabelled = ttools.regress_action(args[0], tuple(args[1:]))
    elif action.kind == ttools.LINGAM_DIR:
        relabelled = ttools.Action(ttools.LINGAM_DIR, tuple(sorted(args)))
    else:
        relabelled = ttools.Action(action.kind, tuple(args))
    return relabelled


def _evidence(x: np.ndarray, actions: List[ttools.Action]) -> ttools.Evidence:
    """
    Record tool calls on a dataset.

    :param x: training rows
    :param actions: calls to make
    :return: evidence after the calls
    """
    evidence = ttools.Evidence(ttools.ToolCache(x), budget=len(actions) + 5)
    for action in actions:
        evidence.record(action)
    return evidence


class Test_featurize(hunitest.TestCase):
    """
    Check the observation built from evidence.
    """

    def test1(self) -> None:
        """
        Relabelling the variables permutes the features and changes nothing else.
        """
        inst = tenv.sample_instance(tenv.GraphConfig(family="linear_exp"), 21)
        x = inst.d_train.to_numpy()
        rng = np.random.default_rng(0)
        all_actions = ttools.enumerate_tool_actions(6)
        picked = rng.choice(len(all_actions), size=60, replace=False)
        actions = [all_actions[k] for k in picked]
        obs = tfeat.featurize(_evidence(x, actions))
        perm = [3, 0, 5, 1, 4, 2]
        new_index = list(np.argsort(perm))
        x_perm = x[:, perm]
        actions_perm = [_relabel(a, new_index) for a in actions]
        obs_perm = tfeat.featurize(_evidence(x_perm, actions_perm))
        expected = tfeat.permute_observation(obs, perm)
        np.testing.assert_allclose(obs_perm.node, expected.node, atol=1e-6)
        np.testing.assert_allclose(obs_perm.pair, expected.pair, atol=1e-6)
        np.testing.assert_allclose(obs_perm.glob, expected.glob, atol=1e-6)

    def test2(self) -> None:
        """
        Directional pair features are antisymmetric; symmetric ones symmetric.
        """
        inst = tenv.sample_instance(tenv.GraphConfig(family="linear_exp"), 22)
        x = inst.d_train.to_numpy()
        actions = [a for a in ttools.enumerate_tool_actions(6)]
        obs = tfeat.featurize(_evidence(x, actions))
        pair = obs.pair
        for name in ("lingam_dir", "proto_dir"):
            k = tfeat.PAIR_FEATURES.index(name)
            np.testing.assert_allclose(
                pair[:, :, k], -pair[:, :, k].T, atol=1e-6
            )
        for name in ("abs_r", "r", "fisher_z", "ci_mask", "min_abs_pcorr"):
            k = tfeat.PAIR_FEATURES.index(name)
            np.testing.assert_allclose(pair[:, :, k], pair[:, :, k].T, atol=1e-6)
        # With every call made, all coverage fractions are 1 off the diagonal.
        off_diag = ~np.eye(6, dtype=bool)
        for name in (
            "ci_cover_1",
            "ci_cover_2",
            "regress_fwd_mask",
            "lingam_mask",
        ):
            k = tfeat.PAIR_FEATURES.index(name)
            self.assertTrue(np.all(pair[:, :, k][off_diag] == 1.0))

    def test3(self) -> None:
        """
        Unmeasured tools are zero with a zero mask; measured ones set the mask.
        """
        inst = tenv.sample_instance(tenv.GraphConfig(family="linear_exp"), 23)
        x = inst.d_train.to_numpy()
        empty = tfeat.featurize(_evidence(x, []))
        tool_columns = [
            k
            for k, name in enumerate(tfeat.PAIR_FEATURES)
            if name not in ("abs_r", "r", "fisher_z", "triad_abs_r")
        ]
        self.assertEqual(
            float(np.abs(empty.pair[:, :, tool_columns]).sum()), 0.0
        )
        action = ttools.Action(ttools.LINGAM_DIR, (1, 4))
        obs = tfeat.featurize(_evidence(x, [action]))
        k_mask = tfeat.PAIR_FEATURES.index("lingam_mask")
        self.assertEqual(
            (
                obs.pair[1, 4, k_mask],
                obs.pair[4, 1, k_mask],
                obs.pair[0, 1, k_mask],
            ),
            (1.0, 1.0, 0.0),
        )

    def test4(self) -> None:
        """
        Feature widths do not depend on the number of variables.
        """
        lines = []
        for n_nodes in (4, 8, 12):
            cfg = tenv.GraphConfig(
                family="mixed", n_nodes=n_nodes, n_edges=n_nodes
            )
            x = tenv.sample_instance(cfg, 24).d_train.to_numpy()
            obs = tfeat.featurize(_evidence(x, []))
            lines.append(f"{obs.node.shape} {obs.pair.shape} {obs.glob.shape}")
        actual = "\n".join(lines)
        expected = """
        (4, 12) (4, 4, 20) (10,)
        (8, 12) (8, 8, 20) (10,)
        (12, 12) (12, 12, 20) (10,)
        """
        self.assert_equal(actual, expected, dedent=True)
