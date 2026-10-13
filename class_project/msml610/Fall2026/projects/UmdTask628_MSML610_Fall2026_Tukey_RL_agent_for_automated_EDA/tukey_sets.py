"""
Evaluation sets: disjoint seed ranges, task distributions and manifests.

Each set is a block of instance seeds inside its own range, so no training
seed can coincide with an evaluation seed. A manifest lists every instance's
generator settings and a content hash of its data and graph. Manifests are
written before any training; evaluation refuses an instance whose hash no
longer matches (the generator changed). Sealed sets (the confirmatory test
sets) can only be loaded by the confirmatory evaluation.

Hashes are computed inside the project's Docker image; numbers are rounded to
10 decimals first so that last-bit differences cannot change them.

Import as:

import tukey_sets as tsets
"""

import dataclasses
import functools
import hashlib
import json
import logging
import os
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

import helpers.hdbg as hdbg
import tukey_env as tenv

_LOG = logging.getLogger(__name__)

# Training draws instance seeds in [0, TRAIN_SEED_END).
TRAIN_SEED_END = 1_000_000_000
# First seed of each evaluation range; ranges are 1e9 wide and disjoint.
SEED_RANGES = {
    "validation": 1_000_000_000,
    "m2": 2_000_000_000,
    "test_main": 3_000_000_000,
    "test_proto": 4_000_000_000,
    "test_reserve": 5_000_000_000,
    "m4": 6_000_000_000,
}
_RANGE_WIDTH = 1_000_000_000
# Seed-stream tag of the size draw, separate from the instance's own streams.
_SIZE_STREAM = 17


# #############################################################################
# Task distributions
# #############################################################################


def draw_size(
    seed: int, *, n_nodes_range: Tuple[int, int] = (6, 8)
) -> Tuple[int, int]:
    """
    Draw a graph size: d uniform in a range, `round(rho * d)` edges.

    :param seed: instance seed
    :param n_nodes_range: smallest and largest d (inclusive)
    :return: number of nodes and of edges (rho ~ U[1, 1.5])
    """
    rng = np.random.default_rng([seed, _SIZE_STREAM])
    n_nodes = int(rng.integers(n_nodes_range[0], n_nodes_range[1] + 1))
    n_edges = int(round(rng.uniform(1.0, 1.5) * n_nodes))
    return n_nodes, n_edges


def main_task_config(seed: int, *, family: str = "mixed") -> tenv.GraphConfig:
    """
    Settings of one instance of the main task distribution.

    d in {6, 7, 8}, `round(rho * d)` edges with rho ~ U[1, 1.5], linear,
    noise family drawn per graph (`mixed`), standardized with train
    statistics. `family` selects one noise family instead.

    :param seed: instance seed
    :param family: generator family
    :return: generator settings
    """
    n_nodes, n_edges = draw_size(seed)
    if family == "linear_gauss_eqvar_raw":
        cfg = dataclasses.replace(
            tenv.EQVAR_RAW_CONFIG, n_nodes=n_nodes, n_edges=n_edges
        )
    else:
        cfg = tenv.GraphConfig(n_nodes=n_nodes, n_edges=n_edges, family=family)
    return cfg


def fixed_size_config(seed: int, *, n_nodes: int) -> tenv.GraphConfig:
    """
    Settings for graphs of one size (d fixed, rho ~ U[1, 1.5], `mixed`).

    :param seed: instance seed
    :param n_nodes: number of nodes
    :return: generator settings
    """
    _, n_edges = draw_size(seed, n_nodes_range=(n_nodes, n_nodes))
    cfg = tenv.GraphConfig(n_nodes=n_nodes, n_edges=n_edges, family="mixed")
    return cfg


def _constant_config(seed: int, *, cfg: tenv.GraphConfig) -> tenv.GraphConfig:
    """
    Return the same settings for every seed.

    :param seed: instance seed (unused)
    :param cfg: settings
    :return: `cfg`
    """
    _ = seed
    return cfg


TASKS: Dict[str, Callable[[int], tenv.GraphConfig]] = {
    "main": main_task_config,
    "main_linear_exp": functools.partial(main_task_config, family="linear_exp"),
    "main_linear_uniform": functools.partial(
        main_task_config, family="linear_uniform"
    ),
    "main_linear_gauss": functools.partial(
        main_task_config, family="linear_gauss"
    ),
    "main_eqvar_raw": functools.partial(
        main_task_config, family="linear_gauss_eqvar_raw"
    ),
    "proto": functools.partial(_constant_config, cfg=tenv.PROTO_CONFIG),
    "mixed_12_18": functools.partial(
        _constant_config,
        cfg=tenv.GraphConfig(n_nodes=12, n_edges=18, family="mixed"),
    ),
    "nonlinear_8_10": functools.partial(
        _constant_config,
        cfg=tenv.GraphConfig(n_nodes=8, n_edges=10, family="nonlinear"),
    ),
    # U, T1, T2 and three unrelated features.
    "confounder_observed_6": functools.partial(
        _constant_config,
        cfg=tenv.GraphConfig(
            n_nodes=6,
            n_edges=2,
            family="sklearn_confounder",
            confounder="observed",
        ),
    ),
    "main_pgmpy": functools.partial(main_task_config, family="pgmpy_lg"),
}
for _n_nodes in (4, 5, 6, 7, 8, 9, 10, 12, 15):
    TASKS[f"size_{_n_nodes}"] = functools.partial(
        fixed_size_config, n_nodes=_n_nodes
    )


# #############################################################################
# Evaluation sets
# #############################################################################


@dataclasses.dataclass(frozen=True)
class EvalSet:
    """
    A named block of instance seeds drawn from one task distribution.
    """

    name: str
    # Range in `SEED_RANGES` and offset of the block inside it.
    seed_range: str
    offset: int
    n_instances: int
    # Key of `TASKS`.
    task: str
    # Sealed sets are loaded only by the confirmatory evaluation.
    sealed: bool

    @property
    def seeds(self) -> List[int]:
        """
        Instance seeds of the set.
        """
        first = SEED_RANGES[self.seed_range] + self.offset
        return list(range(first, first + self.n_instances))


def _make_sets() -> Dict[str, EvalSet]:
    """
    Define every evaluation set of milestones M2 and M3.

    :return: sets by name
    """
    sets = [
        EvalSet("V_main", "validation", 0, 600, "main", False),
        EvalSet("V_proto", "validation", 1_000_000, 200, "proto", False),
        EvalSet("T_main", "test_main", 0, 600, "main", True),
        EvalSet("T_proto", "test_proto", 0, 200, "proto", True),
        EvalSet("T_reserve", "test_reserve", 0, 600, "main", True),
    ]
    # M2 benchmark settings, each with a validation twin for tuning.
    m2_tasks = [
        ("A0", "proto"),
        ("A1", "main_linear_exp"),
        ("B", "main_linear_gauss"),
        ("Braw", "main_eqvar_raw"),
        ("C", "main_linear_uniform"),
        ("D", "main"),
        ("H", "mixed_12_18"),
        # Added with the M1 families (manifests written before the M2 table).
        ("E", "nonlinear_8_10"),
        ("F", "confounder_observed_6"),
        ("G", "main_pgmpy"),
    ]
    for k, (setting, task) in enumerate(m2_tasks):
        offset = k * 1_000_000
        sets.append(EvalSet(f"B_m2_{setting}", "m2", offset, 200, task, False))
        sets.append(
            EvalSet(
                f"V_m2_{setting}",
                "validation",
                10_000_000 + offset,
                200,
                task,
                False,
            )
        )
    # Validation graphs of one size, to measure the matched budget.
    for n_nodes in (4, 5, 6, 7, 8, 9, 10, 12, 15):
        sets.append(
            EvalSet(
                f"V_size_{n_nodes}",
                "validation",
                50_000_000 + n_nodes * 1_000_000,
                200,
                f"size_{n_nodes}",
                False,
            )
        )
    sets_by_name = {eval_set.name: eval_set for eval_set in sets}
    hdbg.dassert_eq(len(sets_by_name), len(sets))
    return sets_by_name


EVAL_SETS = _make_sets()


# #############################################################################
# Manifests
# #############################################################################


def instance_hash(inst: tenv.Instance) -> str:
    """
    Hash an instance's data and graph.

    :param inst: instance
    :return: SHA-256 hex digest of the rounded data, graph and weights
    """
    digest = hashlib.sha256()
    arrays = (
        inst.d_train.to_numpy(),
        inst.d_test.to_numpy(),
        inst.truth.B.astype(float),
        inst.W,
    )
    for array in arrays:
        rounded = np.round(np.ascontiguousarray(array, dtype=float), 10)
        # Map -0.0 to 0.0 so the sign of zero cannot change the bytes.
        digest.update((rounded + 0.0).tobytes())
    return digest.hexdigest()


def build_manifest(eval_set: EvalSet) -> Dict[str, Any]:
    """
    Sample every instance of a set and record its settings and hash.

    :param eval_set: set definition
    :return: manifest dictionary
    """
    instances = []
    for seed in eval_set.seeds:
        cfg = TASKS[eval_set.task](seed)
        inst = tenv.sample_instance(cfg, seed)
        instances.append(
            {
                "seed": seed,
                "n_nodes": cfg.n_nodes,
                "n_edges": cfg.n_edges,
                "family": cfg.family,
                "noise": inst.noise_families[0],
                "target_class": inst.truth.target_class,
                "sha256": instance_hash(inst),
            }
        )
    manifest = {
        "name": eval_set.name,
        "task": eval_set.task,
        "seed_range": eval_set.seed_range,
        "offset": eval_set.offset,
        "n_instances": eval_set.n_instances,
        "sealed": eval_set.sealed,
        "instances": instances,
    }
    return manifest


def manifest_path(manifest_dir: str, name: str) -> str:
    """
    Return the path of a set's manifest.

    :param manifest_dir: directory of the manifests
    :param name: set name
    :return: path of `<name>.json`
    """
    return os.path.join(manifest_dir, f"{name}.json")


def load_set(
    name: str, manifest_dir: str, *, allow_sealed: bool = False
) -> List[Tuple[int, tenv.GraphConfig, str]]:
    """
    Load a set's seeds, settings and expected hashes from its manifest.

    :param name: set name
    :param manifest_dir: directory of the manifests
    :param allow_sealed: True only in the confirmatory evaluation
    :return: `(seed, settings, sha256)` per instance
    """
    eval_set = EVAL_SETS[name]
    if eval_set.sealed:
        hdbg.dassert(
            allow_sealed,
            "Set '%s' is sealed: only the confirmatory evaluation loads it",
            name,
        )
    with open(manifest_path(manifest_dir, name)) as file:
        manifest = json.load(file)
    hdbg.dassert_eq(manifest["n_instances"], eval_set.n_instances)
    items = []
    for item in manifest["instances"]:
        cfg = TASKS[eval_set.task](item["seed"])
        items.append((item["seed"], cfg, item["sha256"]))
    return items


def sample_checked(
    seed: int, cfg: tenv.GraphConfig, expected_hash: str
) -> tenv.Instance:
    """
    Sample a manifest instance and check that the generator did not change.

    :param seed: instance seed
    :param cfg: generator settings
    :param expected_hash: hash recorded in the manifest
    :return: the instance
    """
    inst = tenv.sample_instance(cfg, seed)
    hdbg.dassert_eq(
        instance_hash(inst),
        expected_hash,
        "Instance %s differs from its manifest: the generator changed",
        seed,
    )
    return inst
