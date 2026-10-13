#!/usr/bin/env python

"""
Write evaluation-set manifests, measure the matched tool budget, check gates.

Runs inside the project's Docker image (normally through `run_job.sh`, so the
outputs carry provenance).

# Usage Example

- Write the manifests of every evaluation set (never overwrites):
> tukey_eval.py write_manifests --manifest_dir configs/sets

- Measure the matched budget B_match(d) on the validation size sets:
> tukey_eval.py b_match --manifest_dir configs/sets --out configs/b_match.json

- Apply the learning-signal criteria of gate G7 to a training run:
> tukey_eval.py g7_report --run_dir out --null_run_dir out_null --out g7.json
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


def _paired_bootstrap(
    diff: np.ndarray, *, n_boot: int = 10_000, seed: int = 0
) -> Dict[str, float]:
    """
    Compute the mean of paired differences with a percentile bootstrap CI.

    :param diff: per-graph differences
    :param n_boot: bootstrap resamples
    :param seed: random seed
    :return: mean and 95% interval
    """
    rng = np.random.default_rng(seed)
    idx = rng.integers(len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(axis=1)
    summary = {
        "mean": float(diff.mean()),
        "ci_low": float(np.percentile(means, 2.5)),
        "ci_high": float(np.percentile(means, 97.5)),
    }
    return summary


def _read_evals(run_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    Read the evaluation records of a training run by tag.

    :param run_dir: output directory of `tukey_train.py`
    :return: records by tag (`eval` and `eval_bc` events)
    """
    evals: Dict[str, Dict[str, Any]] = {}
    with open(os.path.join(run_dir, "train_log.jsonl")) as file:
        for line in file:
            record = json.loads(line)
            if record["event"] == "eval":
                evals[record["tag"]] = record
            elif record["event"] == "eval_bc":
                evals["bc_" + record["tag"]] = record
    return evals


def _g7_report(run_dir: str, null_run_dir: str, out: str) -> Dict[str, Any]:
    """
    Apply the spike's learning-signal criteria (gate G7) on validation graphs.

    Criteria (plan 5.1): BC next-action accuracy >= 0.9; BC closed-loop SHD
    within +0.3 of the teacher at the same budget; decoder RL improves the
    validation score over the BC decoder (paired 95% CI above 0); tool RL does
    not lower it by more than 0.1; decoder RL with a null reward does not
    improve it (CI not above 0).

    :param run_dir: run with phases A, B, C
    :param null_run_dir: null-reward run started from the same BC checkpoint
    :param out: output JSON path
    :return: report
    """
    evals = _read_evals(run_dir)
    score = {
        tag: np.array(r["per_graph_score"])
        for tag, r in evals.items()
        if "per_graph_score" in r
    }
    shd = {
        tag: np.array(r["per_graph_shd"])
        for tag, r in evals.items()
        if "per_graph_shd" in r
    }
    report: Dict[str, Any] = {}
    accuracy = evals["bc_after_A"]["accuracy"]
    report["bc_accuracy"] = {"value": accuracy, "pass": accuracy >= 0.9}
    shd_gap = float(
        shd["after_A"].mean() - shd["expert_strength_matched_budget"].mean()
    )
    report["bc_shd_gap"] = {"value": shd_gap, "pass": shd_gap <= 0.3}
    dec = _paired_bootstrap(score["after_B"] - score["after_A"])
    report["decoder_rl_gain"] = {**dec, "pass": dec["ci_low"] > 0}
    tool = _paired_bootstrap(score["after_C"] - score["after_B"])
    report["tool_rl_change"] = {**tool, "pass": tool["mean"] >= -0.1}
    if null_run_dir:
        null_evals = _read_evals(null_run_dir)
        null_diff = np.array(
            null_evals["after_B"]["per_graph_score"]
        ) - np.array(null_evals["start"]["per_graph_score"])
        null = _paired_bootstrap(null_diff)
        report["null_reward_change"] = {**null, "pass": null["ci_low"] <= 0}
    report["all_pass"] = all(
        v["pass"] for v in report.values() if isinstance(v, dict)
    )
    with open(out, "w") as file:
        json.dump(report, file, indent=1)
        file.write("\n")
    _LOG.info("G7 report: %s", report)
    return report


# #############################################################################
# CLI
# #############################################################################


def _parse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=hparser.CustomHelpFormatter,
    )
    parser.add_argument(
        "action",
        choices=["write_manifests", "b_match", "g7_report"],
        help="What to do",
    )
    parser.add_argument(
        "--run_dir", action="store", default="", help="Training run output"
    )
    parser.add_argument(
        "--null_run_dir",
        action="store",
        default="",
        help="Null-reward training run output",
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
    elif args.action == "g7_report":
        hdbg.dassert_ne(args.out, "", "Pass --out")
        _g7_report(args.run_dir, args.null_run_dir, args.out)
    else:
        raise ValueError(f"Unknown action '{args.action}'")


if __name__ == "__main__":
    _main(_parse())
