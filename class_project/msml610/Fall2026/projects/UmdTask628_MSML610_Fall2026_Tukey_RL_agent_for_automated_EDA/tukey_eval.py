#!/usr/bin/env python

"""
Write evaluation-set manifests and measure the matched tool budget.

Runs inside the project's Docker image (normally through `run_job.sh`, so the
outputs carry provenance).

# Usage Example

- Write the manifests of every evaluation set (never overwrites):
> tukey_eval.py write_manifests --manifest_dir configs/sets

- Measure the matched budget B_match(d) on the validation size sets:
> tukey_eval.py b_match --manifest_dir configs/sets --out configs/b_match.json
"""

import argparse
import json
import logging
import os
from typing import Any, Dict, List

import numpy as np

import helpers.hdbg as hdbg
import helpers.hparser as hparser
import tukey_env as tenv
import tukey_scripted as tscrip
import tukey_sets as tsets

_LOG = logging.getLogger(__name__)

# Budget large enough never to bind (all tool calls on 15 nodes).
_UNLIMITED_BUDGET = 1_000_000


# #############################################################################
# Actions
# #############################################################################


def _write_manifests(manifest_dir: str, names: List[str]) -> None:
    """
    Write the manifest of each set, refusing to overwrite an existing one.

    :param manifest_dir: output directory
    :param names: set names; empty means every set
    """
    names = names or sorted(tsets.EVAL_SETS)
    os.makedirs(manifest_dir, exist_ok=True)
    for name in names:
        path = tsets.manifest_path(manifest_dir, name)
        hdbg.dassert(not os.path.exists(path), "Manifest exists: %s", path)
        manifest = tsets.build_manifest(tsets.EVAL_SETS[name])
        with open(path, "w") as file:
            json.dump(manifest, file, indent=1)
            file.write("\n")
        _LOG.info("Wrote %s (%s instances)", path, manifest["n_instances"])


def _measure_b_match(manifest_dir: str, out: str) -> Dict[str, Any]:
    """
    Count the scripted expert's calls on the validation size sets.

    B_match(d) is the median number of calls the unbudgeted prototype rule
    makes on validation graphs with d nodes.

    :param manifest_dir: directory of the manifests
    :param out: output JSON path
    :return: statistics per d
    """
    expert = tscrip.ScriptedExpert()
    stats: Dict[str, Any] = {}
    for name in sorted(tsets.EVAL_SETS):
        if not name.startswith("V_size_"):
            continue
        calls = []
        shd = []
        for seed, cfg, sha256 in tsets.load_set(name, manifest_dir):
            tsets.sample_checked(seed, cfg, sha256)
            env = tenv.TukeyEnv(cfg, budget=_UNLIMITED_BUDGET)
            scores, history = tscrip.run_episode(env, expert, seed=seed)
            calls.append(len(history))
            shd.append(scores["shd_target"])
        n_nodes = int(name.split("_")[-1])
        stats[str(n_nodes)] = {
            "set": name,
            "n_graphs": len(calls),
            "b_match": float(np.median(calls)),
            "calls_q25": float(np.percentile(calls, 25)),
            "calls_q75": float(np.percentile(calls, 75)),
            "calls_max": int(np.max(calls)),
            "expert_shd_target_mean": float(np.mean(shd)),
        }
        _LOG.info("d=%s: %s", n_nodes, stats[str(n_nodes)])
    with open(out, "w") as file:
        json.dump(stats, file, indent=1)
        file.write("\n")
    return stats


# #############################################################################
# CLI
# #############################################################################


def _parse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=hparser.CustomHelpFormatter,
    )
    parser.add_argument(
        "action", choices=["write_manifests", "b_match"], help="What to do"
    )
    parser.add_argument(
        "--manifest_dir",
        action="store",
        default="configs/sets",
        help="Directory of the set manifests",
    )
    parser.add_argument(
        "--names",
        action="store",
        nargs="*",
        default=[],
        help="Sets to write (default: all)",
    )
    parser.add_argument(
        "--out", action="store", default="", help="Output JSON path"
    )
    hparser.add_verbosity_arg(parser)
    return parser


def _main(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args()
    hdbg.init_logger(verbosity=args.log_level, use_exec_path=True)
    if args.action == "write_manifests":
        _write_manifests(args.manifest_dir, args.names)
    elif args.action == "b_match":
        hdbg.dassert_ne(args.out, "", "Pass --out")
        _measure_b_match(args.manifest_dir, args.out)
    else:
        raise ValueError(f"Unknown action '{args.action}'")


if __name__ == "__main__":
    _main(_parse())
