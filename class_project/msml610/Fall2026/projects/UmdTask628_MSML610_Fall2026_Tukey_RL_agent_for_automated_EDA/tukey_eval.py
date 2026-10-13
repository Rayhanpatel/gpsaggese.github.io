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

- Break a BC policy's agreement with its teacher down by decision type:
> tukey_eval.py bc_diagnostics --checkpoint out/policy_after_A.pt --out d.json

- Apply the learning-signal criteria of gate G7 to a training run:
> tukey_eval.py g7_report --bc_run_dir bc --rl_run_dir rl --null_run_dir null \
    --out g7.json
"""

import argparse
import json
import logging
import os
from typing import Any, Dict, List

import numpy as np
import torch

import helpers.hdbg as hdbg
import helpers.hparser as hparser
import tukey_env as tenv
import tukey_features as tfeat
import tukey_policy as tpolic
import tukey_rollout as troll
import tukey_scripted as tscrip
import tukey_sets as tsets
import tukey_tools as ttools

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


def _bc_diagnostics(
    checkpoint: str, manifest_dir: str, b_match_path: str, out: str
) -> Dict[str, Any]:
    """
    Break the policy's next-action agreement with its teacher down by type.

    Replays the teacher on the first 100 `V_main` graphs with the budgets of
    `tukey_train.py`'s accuracy check and classifies each teacher decision:
    `ci_start` (first CI test of a pair), `ci_continue` (another conditioning
    variable for a pair already tested), `regress` (orientation) or `stop`.
    For mistakes it records what the policy chose instead and, when both
    choices are CI tests, how close they were on the teacher's own ordering
    score (`|r|` of the pair, or `max(|r_ik|, |r_jk|)` of the conditioning
    variable).

    :param checkpoint: saved policy (`tukey_train.py`)
    :param manifest_dir: directory of the manifests
    :param b_match_path: matched budgets
    :param out: output JSON path
    :return: diagnostics
    """
    state = torch.load(checkpoint, weights_only=False)
    hdbg.dassert_eq(state["feature_schema"], tfeat.schema_hash())
    policy = tpolic.TukeyPolicy(hidden=state["config"]["hidden"])
    policy.load_state_dict(state["state_dict"])
    teacher = tscrip.ScriptedExpert(order=state["config"]["teacher_order"])
    with open(b_match_path) as file:
        b_match = {int(d): v["b_match"] for d, v in json.load(file).items()}
    rng = np.random.default_rng(12345)
    tasks = []
    for seed, cfg, sha256 in tsets.load_set("V_main", manifest_dir)[:100]:
        tsets.sample_checked(seed, cfg, sha256)
        budget = int(round(rng.uniform() * b_match[cfg.n_nodes]))
        tasks.append(troll.Task(cfg, seed, budget))
    episodes = troll.run_expert_episodes(teacher, policy, tasks)
    abs_r = tfeat.PAIR_FEATURES.index("abs_r")
    ci_mask = tfeat.PAIR_FEATURES.index("ci_mask")
    stats: Dict[str, Dict[str, Any]] = {}
    for episode in episodes:
        actions, _ = policy.candidates(episode.task.cfg.n_nodes)
        for decision in episode.decisions:
            with torch.no_grad():
                log_probs = policy.tool_log_probs(decision.obs, decision.legal)
            chosen = actions[int(torch.argmax(log_probs))]
            target = actions[decision.action_index]
            pair = decision.obs.pair
            if target.kind == ttools.CI_TEST:
                i, j = target.args[:2]
                kind = "ci_continue" if pair[i, j, ci_mask] > 0 else "ci_start"
            else:
                kind = target.kind
            entry = stats.setdefault(
                kind, {"n": 0, "correct": 0, "chose": {}, "score_gaps": []}
            )
            entry["n"] += 1
            if chosen == target:
                entry["correct"] += 1
                continue
            if chosen.kind == ttools.CI_TEST and target.kind == ttools.CI_TEST:
                same_pair = chosen.args[:2] == target.args[:2]
                label = "ci_same_pair" if same_pair else "ci_other_pair"
                if same_pair:
                    i, j = target.args[:2]
                    k_t, k_c = target.args[2:], chosen.args[2:]
                    gap = max(
                        pair[i, k_t[0], abs_r], pair[j, k_t[0], abs_r]
                    ) - max(pair[i, k_c[0], abs_r], pair[j, k_c[0], abs_r])
                else:
                    gap = (
                        pair[target.args[0], target.args[1], abs_r]
                        - pair[chosen.args[0], chosen.args[1], abs_r]
                    )
                entry["score_gaps"].append(float(gap))
            else:
                label = chosen.kind
            entry["chose"][label] = entry["chose"].get(label, 0) + 1
    report: Dict[str, Any] = {}
    for kind, entry in sorted(stats.items()):
        gaps = np.abs(entry["score_gaps"])
        report[kind] = {
            "n": entry["n"],
            "accuracy": round(entry["correct"] / entry["n"], 4),
            "mistakes_chose": entry["chose"],
            "median_abs_score_gap": float(np.median(gaps))
            if len(gaps)
            else None,
            "frac_gap_below_0.02": float(np.mean(gaps < 0.02))
            if len(gaps)
            else None,
        }
    n_total = sum(e["n"] for e in stats.values())
    report["overall_accuracy"] = round(
        sum(e["correct"] for e in stats.values()) / n_total, 4
    )
    with open(out, "w") as file:
        json.dump(report, file, indent=1)
        file.write("\n")
    _LOG.info("BC diagnostics: %s", report)
    return report


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


def _g7_report(
    bc_run_dir: str, rl_run_dir: str, null_run_dir: str, out: str
) -> Dict[str, Any]:
    """
    Apply the spike's learning-signal criteria (gate G7) on validation graphs.

    Criteria (plan 5.1): BC next-action accuracy >= 0.9; BC closed-loop SHD
    within +0.3 of the teacher at the same budget; decoder RL improves the
    validation score over the BC decoder (paired 95% CI above 0); tool RL does
    not lower it by more than 0.1; decoder RL with a null reward does not
    improve it (CI not above 0). All runs evaluate the same validation graphs.

    :param bc_run_dir: behaviour-cloning run (phase A)
    :param rl_run_dir: run with phases B then C started from the BC checkpoint
        (its "start" evaluation is the BC policy)
    :param null_run_dir: null-reward phase-B run from the same checkpoint
    :param out: output JSON path
    :return: report
    """
    bc = _read_evals(bc_run_dir)
    rl = _read_evals(rl_run_dir)

    def _scores(evals: Dict[str, Dict[str, Any]], tag: str) -> np.ndarray:
        return np.array(evals[tag]["per_graph_score"])

    report: Dict[str, Any] = {}
    accuracy = bc["bc_after_A"]["accuracy"]
    report["bc_accuracy"] = {"value": accuracy, "pass": accuracy >= 0.9}
    shd_gap = float(
        np.mean(bc["after_A"]["per_graph_shd"])
        - np.mean(bc["expert_strength_matched_budget"]["per_graph_shd"])
    )
    report["bc_shd_gap"] = {"value": shd_gap, "pass": shd_gap <= 0.3}
    dec = _paired_bootstrap(_scores(rl, "after_B") - _scores(rl, "start"))
    report["decoder_rl_gain"] = {**dec, "pass": dec["ci_low"] > 0}
    tool = _paired_bootstrap(_scores(rl, "after_C") - _scores(rl, "after_B"))
    report["tool_rl_change"] = {**tool, "pass": tool["mean"] >= -0.1}
    null_evals = _read_evals(null_run_dir)
    null = _paired_bootstrap(
        _scores(null_evals, "after_B") - _scores(null_evals, "start")
    )
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
        choices=["write_manifests", "b_match", "g7_report", "bc_diagnostics"],
        help="What to do",
    )
    parser.add_argument(
        "--bc_run_dir", action="store", default="", help="BC run output"
    )
    parser.add_argument(
        "--checkpoint", action="store", default="", help="Saved policy"
    )
    parser.add_argument(
        "--b_match_path",
        action="store",
        default="configs/b_match.json",
        help="Matched budgets",
    )
    parser.add_argument(
        "--rl_run_dir",
        action="store",
        default="",
        help="Decoder and tool RL run output",
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
    elif args.action == "bc_diagnostics":
        hdbg.dassert_ne(args.out, "", "Pass --out")
        _bc_diagnostics(
            args.checkpoint, args.manifest_dir, args.b_match_path, args.out
        )
    elif args.action == "g7_report":
        hdbg.dassert_ne(args.out, "", "Pass --out")
        _g7_report(args.bc_run_dir, args.rl_run_dir, args.null_run_dir, args.out)
    else:
        raise ValueError(f"Unknown action '{args.action}'")


if __name__ == "__main__":
    _main(_parse())
