"""
Test the evaluation sets: disjoint seeds, task distributions and manifests.

Import as:

import test.test_tukey_sets as ttetuset
"""

import collections
import json
import logging
import os

import helpers.hunit_test as hunitest
import tukey_env as tenv
import tukey_sets as tsets

_LOG = logging.getLogger(__name__)


class Test_eval_sets(hunitest.TestCase):
    """
    Seeds of different sets never collide, nor with training seeds.
    """

    def test1(self) -> None:
        """
        All set seeds are distinct and outside the training range.
        """
        seen = set()
        n_seeds = 0
        for eval_set in tsets.EVAL_SETS.values():
            seeds = eval_set.seeds
            self.assertGreaterEqual(min(seeds), tsets.TRAIN_SEED_END)
            seen.update(seeds)
            n_seeds += len(seeds)
        self.assertEqual(len(seen), n_seeds)

    def test2(self) -> None:
        """
        Exactly the three confirmatory test sets are sealed.
        """
        sealed = sorted(n for n, s in tsets.EVAL_SETS.items() if s.sealed)
        self.assertEqual(sealed, ["T_main", "T_proto", "T_reserve"])


class Test_main_task_config(hunitest.TestCase):
    """
    The main task distribution draws d in {6, 7, 8} and 1-1.5 edges per node.
    """

    def test1(self) -> None:
        """
        Sizes and edge counts over 3,000 seeds.
        """
        sizes: collections.Counter = collections.Counter()
        for seed in range(3000):
            cfg = tsets.main_task_config(seed)
            sizes[cfg.n_nodes] += 1
            self.assertGreaterEqual(cfg.n_edges, cfg.n_nodes)
            self.assertLessEqual(cfg.n_edges, round(1.5 * cfg.n_nodes))
            self.assertEqual(cfg.family, "mixed")
        self.assertEqual(sorted(sizes), [6, 7, 8])
        self.assertGreater(min(sizes.values()), 900)

    def test2(self) -> None:
        """
        A family override keeps the size and the eqvar settings are enforced.
        """
        cfg = tsets.main_task_config(5)
        cfg_raw = tsets.main_task_config(5, family="linear_gauss_eqvar_raw")
        self.assertEqual(
            (cfg.n_nodes, cfg.n_edges), (cfg_raw.n_nodes, cfg_raw.n_edges)
        )
        self.assertEqual(
            (cfg_raw.noise_scale, cfg_raw.standardize), ("equal", "none")
        )


class Test_manifests(hunitest.TestCase):
    """
    Manifests round-trip, detect generator changes and guard sealed sets.
    """

    def test1(self) -> None:
        """
        A written manifest loads back and its instances pass the hash check.
        """
        manifest_dir = self.get_scratch_space()
        manifest = tsets.build_manifest(tsets.EVAL_SETS["V_proto"])
        path = tsets.manifest_path(manifest_dir, "V_proto")
        with open(path, "w") as file:
            json.dump(manifest, file)
        items = tsets.load_set("V_proto", manifest_dir)
        self.assertEqual(len(items), 200)
        for seed, cfg, sha256 in items[:3]:
            inst = tsets.sample_checked(seed, cfg, sha256)
            self.assertEqual(inst.cfg, tenv.PROTO_CONFIG)

    def test2(self) -> None:
        """
        The hash is stable across samples and changes with the data.
        """
        cfg = tsets.main_task_config(12)
        hash1 = tsets.instance_hash(tenv.sample_instance(cfg, 12))
        hash2 = tsets.instance_hash(tenv.sample_instance(cfg, 12))
        self.assertEqual(hash1, hash2)
        with self.assertRaises(AssertionError):
            tsets.sample_checked(13, tsets.main_task_config(13), hash1)

    def test3(self) -> None:
        """
        Sealed sets refuse to load unless the caller is the confirmatory run.
        """
        manifest_dir = self.get_scratch_space()
        with self.assertRaises(AssertionError):
            tsets.load_set("T_main", manifest_dir)
        self.assertFalse(
            os.path.exists(tsets.manifest_path(manifest_dir, "T_main"))
        )
