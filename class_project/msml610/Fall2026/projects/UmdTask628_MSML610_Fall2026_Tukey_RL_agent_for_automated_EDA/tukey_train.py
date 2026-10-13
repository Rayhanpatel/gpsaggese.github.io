#!/usr/bin/env python

"""
Train the Tukey policy: behaviour cloning (A), decoder RL (B), tool RL (C).

Every phase draws fresh training graphs from the main task distribution
(training seed range only) with tool budgets drawn uniformly between 0 and
the matched budget B_match(d). Progress is measured on validation graphs
(never on a test set) at the matched budget, against the scripted expert at
the same budget. Each iteration writes one JSON line with losses, rewards and
timings to `<out_dir>/train_log.jsonl`; checkpoints and the final evaluation
go to the same directory.

# Usage Example

- Integration-spike run (minutes):
> tukey_train.py --preset spike --out_dir runs_out/spike_seed0 --seed 0

- Null-reward control (decoder RL with shuffled rewards from a BC checkpoint):
> tukey_train.py --preset spike_null --out_dir runs_out/null --seed 0 \
    --init_checkpoint runs_out/spike_seed0/policy_after_A.pt
"""

import argparse
import copy
import dataclasses
import json
import logging
import os
import resource
import time
from typing import Any, Dict, List

import numpy as np
import torch

import helpers.hdbg as hdbg
import helpers.hparser as hparser
import tukey_features as tfeat
import tukey_policy as tpolic
import tukey_rl as trl
import tukey_rollout as troll
import tukey_scripted as tscrip
import tukey_sets as tsets

_LOG = logging.getLogger(__name__)


# #############################################################################
# Configuration
# #############################################################################


@dataclasses.dataclass
class TrainConfig:
    """
    Hyperparameters of one training run.
    """

    seed: int = 0
    hidden: int = 64
    phases: str = "ABC"
    # Processing order of the behaviour-cloning teacher (see `ScriptedExpert`).
    teacher_order: str = "strength"
    # Phase A: behaviour cloning of the budgeted scripted expert.
    bc_iters: int = 60
    bc_graphs: int = 16
    bc_lr: float = 1e-3
    # Phase B: decoder RL with RLOO on states from the BC tool head.
    dec_iters: int = 40
    dec_graphs: int = 32
    dec_samples: int = 8
    dec_eps_random: float = 0.5
    dec_lr: float = 1e-3
    # Phase C: tool RL with the decoder frozen.
    tool_iters: int = 10
    tool_graphs: int = 8
    tool_rollouts: int = 8
    tool_samples: int = 8
    tool_lr: float = 3e-4
    cost: float = 0.001
    shaping: bool = True
    entropy_coef: float = 0.01
    value_coef: float = 1.0
    # Control: replace decoder rewards by a random permutation (no signal).
    null_reward: bool = False
    # Evaluation on validation graphs at the matched budget.
    eval_set: str = "V_main"
    eval_graphs: int = 100
    eval_every: int = 0
    # Inputs.
    b_match_path: str = "configs/b_match.json"
    manifest_dir: str = "configs/sets"


PRESETS: Dict[str, Dict[str, Any]] = {
    "spike": {},
    "spike_null": {"phases": "B", "null_reward": True},
}


# #############################################################################
# Helpers
# #############################################################################


def _rss_mb() -> float:
    """
    Return the peak resident memory of this process in MB.
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _null_score_fn(rng: np.random.Generator):
    """
    Return a score function that ignores the graph (uniform noise in [-1, 0]).
    """

    def _score(B: np.ndarray) -> float:
        _ = B
        return float(-rng.uniform())

    return _score


class _Trainer:
    """
    Hold the policy, the data sources and the log of one run.
    """

    def __init__(self, cfg: TrainConfig, out_dir: str) -> None:
        self.cfg = cfg
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        torch.manual_seed(cfg.seed)
        self.rng = np.random.default_rng(cfg.seed)
        self.generator = torch.Generator().manual_seed(cfg.seed)
        self.policy = tpolic.TukeyPolicy(hidden=cfg.hidden)
        self.expert = tscrip.ScriptedExpert(order=cfg.teacher_order)
        with open(cfg.b_match_path) as file:
            self.b_match = {
                int(d): v["b_match"] for d, v in json.load(file).items()
            }
        items = tsets.load_set(cfg.eval_set, cfg.manifest_dir)[: cfg.eval_graphs]
        self.eval_tasks = []
        for seed, graph_cfg, sha256 in items:
            tsets.sample_checked(seed, graph_cfg, sha256)
            budget = int(round(self.b_match[graph_cfg.n_nodes]))
            self.eval_tasks.append(troll.Task(graph_cfg, seed, budget))
        self._log_file = open(os.path.join(out_dir, "train_log.jsonl"), "a")

    def log(self, record: Dict[str, Any]) -> None:
        """
        Append one JSON line to the training log.
        """
        record["rss_mb"] = round(_rss_mb(), 1)
        self._log_file.write(json.dumps(record) + "\n")
        self._log_file.flush()
        _LOG.info(record)

    def training_tasks(self, n_graphs: int) -> List[troll.Task]:
        """
        Draw fresh training graphs with budgets U[0, 1] x B_match(d).
        """
        tasks = []
        for _ in range(n_graphs):
            seed = int(self.rng.integers(tsets.TRAIN_SEED_END))
            graph_cfg = tsets.main_task_config(seed)
            frac = self.rng.uniform()
            budget = int(round(frac * self.b_match[graph_cfg.n_nodes]))
            tasks.append(troll.Task(graph_cfg, seed, budget))
        return tasks

    def evaluate(self, tag: str) -> Dict[str, Any]:
        """
        Score the greedy policy on the validation graphs at the matched budget.
        """
        start = time.time()
        results = troll.evaluate_policy(
            self.policy, self.eval_tasks, generator=self.generator
        )
        summary = {
            "event": "eval",
            "tag": tag,
            "score": float(np.mean([r["score"] for r in results])),
            "shd_target": float(np.mean([r["shd_target"] for r in results])),
            "calls": float(np.mean([r["calls_used"] for r in results])),
            "per_graph_score": [round(r["score"], 6) for r in results],
            "per_graph_shd": [int(r["shd_target"]) for r in results],
            "seconds": round(time.time() - start, 2),
        }
        self.log(summary)
        return summary

    def evaluate_bc_accuracy(self, tag: str) -> Dict[str, Any]:
        """
        Measure next-action agreement with the teacher on validation graphs.

        Budgets are drawn as in training (U[0, 1] x B_match(d)) from a fixed
        generator, so every evaluation scores the same decisions.
        """
        rng = np.random.default_rng(12345)
        tasks = []
        for task in self.eval_tasks:
            budget = rng.uniform() * self.b_match[task.cfg.n_nodes]
            tasks.append(troll.Task(task.cfg, task.seed, int(round(budget))))
        episodes = troll.run_expert_episodes(self.expert, self.policy, tasks)
        decisions = [d for e in episodes for d in e.decisions]
        summary = {
            "event": "eval_bc",
            "tag": tag,
            "accuracy": round(trl.bc_accuracy(self.policy, decisions), 4),
            "n_decisions": len(decisions),
        }
        self.log(summary)
        return summary

    def evaluate_expert(self, order: str) -> Dict[str, Any]:
        """
        Score a scripted expert on the validation graphs at the same budget.

        :param order: the expert's processing order
        """
        expert = tscrip.ScriptedExpert(order=order)
        results = troll.evaluate_expert(expert, self.policy, self.eval_tasks)
        summary = {
            "event": "eval",
            "tag": f"expert_{order}_matched_budget",
            "score": float(np.mean([r["score"] for r in results])),
            "shd_target": float(np.mean([r["shd_target"] for r in results])),
            "calls": float(np.mean([r["calls_used"] for r in results])),
            "per_graph_score": [round(r["score"], 6) for r in results],
            "per_graph_shd": [int(r["shd_target"]) for r in results],
        }
        self.log(summary)
        return summary

    def save(self, name: str) -> None:
        """
        Save the policy with its configuration and feature schema.
        """
        path = os.path.join(self.out_dir, name)
        torch.save(
            {
                "state_dict": self.policy.state_dict(),
                "config": dataclasses.asdict(self.cfg),
                "feature_schema": tfeat.schema_hash(),
            },
            path,
        )

    def close(self) -> None:
        """
        Close the training log.
        """
        self._log_file.close()

    # ##########################################################################
    # Phases
    # ##########################################################################

    def phase_a(self) -> None:
        """
        Behaviour cloning of the tool choices and the emitted DAG.
        """
        cfg = self.cfg
        optimizer = torch.optim.Adam(self.policy.parameters(), lr=cfg.bc_lr)
        for it in range(cfg.bc_iters):
            start = time.time()
            episodes = troll.run_expert_episodes(
                self.expert, self.policy, self.training_tasks(cfg.bc_graphs)
            )
            t_data = time.time() - start
            decisions = [d for e in episodes for d in e.decisions]
            finals = []
            for episode in episodes:
                n_nodes = episode.task.cfg.n_nodes
                B = np.zeros((n_nodes, n_nodes), dtype=int)
                for parent, child in self.expert.emit_graph(
                    episode.final_evidence
                ):
                    B[parent, child] = 1
                finals.append((episode.final_obs, B))
            optimizer.zero_grad()
            tool_loss, accuracy = float("nan"), float("nan")
            if decisions:
                tool_loss, accuracy = trl.accumulate_bc_tool(
                    self.policy, decisions
                )
            dec_loss = trl.accumulate_bc_decoder(self.policy, finals)
            optimizer.step()
            self.log(
                {
                    "event": "train",
                    "phase": "A",
                    "iter": it,
                    "tool_nll": round(tool_loss, 4),
                    "train_accuracy": round(accuracy, 4),
                    "decoder_nll": round(dec_loss, 4),
                    "n_decisions": len(decisions),
                    "t_data": round(t_data, 3),
                    "t_update": round(time.time() - start - t_data, 3),
                }
            )
            if cfg.eval_every and (it + 1) % cfg.eval_every == 0:
                self.evaluate(f"A_{it + 1}")
                self.evaluate_bc_accuracy(f"A_{it + 1}")

    def phase_b(self) -> None:
        """
        Decoder RL: M sampled graphs per state, exact scores, RLOO.
        """
        cfg = self.cfg
        params = list(self.policy.decoder_encoder.parameters())
        params += list(self.policy.node_score.parameters())
        params += list(self.policy.edge_score.parameters())
        optimizer = torch.optim.Adam(params, lr=cfg.dec_lr)
        for it in range(cfg.dec_iters):
            start = time.time()
            episodes = troll.run_policy_episodes(
                self.policy,
                self.training_tasks(cfg.dec_graphs),
                greedy=False,
                generator=self.generator,
                eps_random=cfg.dec_eps_random,
            )
            t_data = time.time() - start
            states = []
            for episode in episodes:
                score_fn = episode.score_fn
                if cfg.null_reward:
                    score_fn = _null_score_fn(self.rng)
                states.append((episode.final_obs, score_fn))
            optimizer.zero_grad()
            mean_reward = trl.accumulate_decoder_rloo(
                self.policy,
                states,
                n_samples=cfg.dec_samples,
                generator=self.generator,
            )
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            self.log(
                {
                    "event": "train",
                    "phase": "B",
                    "iter": it,
                    "sampled_score": round(mean_reward, 4),
                    "t_data": round(t_data, 3),
                    "t_update": round(time.time() - start - t_data, 3),
                }
            )
            if cfg.eval_every and (it + 1) % cfg.eval_every == 0:
                self.evaluate(f"B_{it + 1}")

    def phase_c(self) -> None:
        """
        Tool RL with the decoder frozen (a copy of the current decoder).
        """
        cfg = self.cfg
        frozen = copy.deepcopy(self.policy)
        params = list(self.policy.tool_encoder.parameters())
        params += list(self.policy.tool_head.parameters())
        params += list(self.policy.value_head.parameters())
        optimizer = torch.optim.Adam(params, lr=cfg.tool_lr)
        for it in range(cfg.tool_iters):
            start = time.time()
            groups = []
            for task in self.training_tasks(cfg.tool_graphs):
                groups.append([task] * cfg.tool_rollouts)
            flat = [task for group in groups for task in group]
            episodes = troll.run_policy_episodes(
                self.policy, flat, greedy=False, generator=self.generator
            )
            episode_groups = [
                episodes[k * cfg.tool_rollouts : (k + 1) * cfg.tool_rollouts]
                for k in range(len(groups))
            ]
            t_data = time.time() - start
            optimizer.zero_grad()
            stats = trl.accumulate_tool_rl(
                self.policy,
                frozen,
                episode_groups,
                cost=cfg.cost,
                shaping=cfg.shaping,
                n_phi_samples=cfg.tool_samples,
                generator=self.generator,
                entropy_coef=cfg.entropy_coef,
                value_coef=cfg.value_coef,
            )
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            record = {"event": "train", "phase": "C", "iter": it}
            record.update({k: round(v, 5) for k, v in stats.items()})
            record["n_decisions"] = sum(len(e.decisions) for e in episodes)
            record["t_data"] = round(t_data, 3)
            record["t_update"] = round(time.time() - start - t_data, 3)
            self.log(record)
            if cfg.eval_every and (it + 1) % cfg.eval_every == 0:
                self.evaluate(f"C_{it + 1}")


# #############################################################################
# CLI
# #############################################################################


def _parse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=hparser.CustomHelpFormatter,
    )
    parser.add_argument(
        "--preset", action="store", default="spike", choices=sorted(PRESETS)
    )
    parser.add_argument("--out_dir", action="store", required=True)
    parser.add_argument("--seed", action="store", type=int, default=0)
    parser.add_argument(
        "--init_checkpoint",
        action="store",
        default="",
        help="Start from a saved policy",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override a config field, e.g. --set bc_iters=200",
    )
    hparser.add_verbosity_arg(parser)
    return parser


def _make_config(preset: str, seed: int, overrides: List[str]) -> TrainConfig:
    """
    Build a configuration from a preset and `key=value` overrides.
    """
    cfg = TrainConfig(seed=seed, **PRESETS[preset])
    for override in overrides:
        key, value = override.split("=", 1)
        hdbg.dassert(hasattr(cfg, key), "Unknown field '%s'", key)
        field_type = type(getattr(cfg, key))
        if field_type is bool:
            parsed: Any = value.lower() in ("1", "true", "yes")
        else:
            parsed = field_type(value)
        setattr(cfg, key, parsed)
    return cfg


def _main(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args()
    hdbg.init_logger(verbosity=args.log_level, use_exec_path=True)
    torch.set_num_threads(1)
    cfg = _make_config(args.preset, args.seed, args.set)
    trainer = _Trainer(cfg, args.out_dir)
    with open(os.path.join(args.out_dir, "config.json"), "w") as file:
        json.dump(dataclasses.asdict(cfg), file, indent=1)
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, weights_only=False)
        hdbg.dassert_eq(checkpoint["feature_schema"], tfeat.schema_hash())
        trainer.policy.load_state_dict(checkpoint["state_dict"])
    trainer.evaluate_expert("index")
    trainer.evaluate_expert("strength")
    trainer.evaluate("start")
    for phase in cfg.phases:
        start = time.time()
        getattr(trainer, f"phase_{phase.lower()}")()
        trainer.log(
            {
                "event": "phase_done",
                "phase": phase,
                "seconds": time.time() - start,
            }
        )
        trainer.save(f"policy_after_{phase}.pt")
        trainer.evaluate(f"after_{phase}")
        if phase == "A":
            trainer.evaluate_bc_accuracy("after_A")
    trainer.close()


if __name__ == "__main__":
    _main(_parse())
