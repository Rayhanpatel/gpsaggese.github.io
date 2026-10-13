"""
Training losses and rollouts: behaviour cloning, decoder RLOO, tool REINFORCE.

The objective everywhere is `J = E[score(G)] - c * E[calls]`, where G is
exactly the graph the decoder emits.

- Phase A (`bc_tool_loss`, `bc_decoder_loss`): imitate the scripted expert
- Phase B (`decoder_rloo_loss`): M graphs per state, each scored exactly, with
  the leave-one-out mean as baseline (RLOO); unbiased for the decoder's
  gradient
- Phase C (`tool_rollout`, `tool_reinforce_loss`): tool head with the decoder
  frozen. With shaping, the return of decision t is
  `Phi(s_T) - Phi(s_t) - c * calls_from_t`, where `Phi(s)` is the frozen
  decoder's expected score if the episode stopped at s, estimated from M
  decoder samples. `Phi(s_t)` is a baseline (it does not depend on the action
  taken at t) and `Phi(s_T)` replaces the sampled terminal score by its
  conditional mean (Rao-Blackwellisation), so the expected gradient is that of
  J and only the variance changes. Without shaping, the return is the sampled
  score of one emitted graph minus the call costs, with a leave-one-out
  baseline over K rollouts on the same instance

Losses take log-probability tensors and plain rewards, so the gradient checks
can feed them exactly enumerated distributions.

Import as:

import tukey_rl as trl
"""

import dataclasses
import logging
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

import helpers.hdbg as hdbg
import tukey_env as tenv
import tukey_features as tfeat
import tukey_policy as tpolic
import tukey_rollout as troll
import tukey_scripted as tscrip
import tukey_tools as ttools

_LOG = logging.getLogger(__name__)

LegalFn = Callable[[List[ttools.Action], ttools.Evidence], np.ndarray]


# #############################################################################
# Phase A: behaviour cloning
# #############################################################################


@dataclasses.dataclass(frozen=True)
class ExpertStep:
    """
    One state of an expert trajectory with the expert's choice.
    """

    obs: tfeat.Observation
    legal: np.ndarray
    # Index of the expert's action in `TukeyPolicy.candidates(d)`.
    action_index: int


def expert_trajectory(
    env: tenv.TukeyEnv,
    expert: tscrip.ScriptedExpert,
    policy: tpolic.TukeyPolicy,
    *,
    seed: int,
) -> Tuple[List[ExpertStep], ttools.Evidence]:
    """
    Record the expert's decisions on one instance.

    :param env: environment (its budget limits the expert)
    :param expert: scripted expert
    :param policy: policy whose candidate list indexes the actions
    :param seed: instance seed
    :return: decision states and the final evidence
    """
    evidence, _ = env.reset(seed=seed)
    actions, _ = policy.candidates(evidence.n_nodes)
    position = {action: k for k, action in enumerate(actions)}
    steps = []
    while evidence.budget_left > 0:
        action = expert.next_action(evidence)
        legal = tpolic.legal_mask(actions, evidence)
        steps.append(
            ExpertStep(tfeat.featurize(evidence), legal, position[action])
        )
        if action.kind == ttools.STOP:
            break
        evidence, _, _, _, _ = env.step(action)
    return steps, evidence


def bc_tool_loss(
    policy: tpolic.TukeyPolicy, steps: List[ExpertStep]
) -> torch.Tensor:
    """
    Compute the mean negative log-likelihood of the expert's tool choices.

    :param policy: policy
    :param steps: expert decision states
    :return: loss
    """
    hdbg.dassert_lt(0, len(steps))
    losses = []
    for step in steps:
        log_probs = policy.tool_log_probs(step.obs, step.legal)
        losses.append(-log_probs[step.action_index])
    loss = torch.stack(losses).mean()
    return loss


def bc_decoder_loss(
    policy: tpolic.TukeyPolicy, obs: tfeat.Observation, B_expert: np.ndarray
) -> torch.Tensor:
    """
    Compute `-log p(G_expert, order)` with the order the decoder prefers.

    The order is the topological order of the expert's DAG with ties broken by
    the current utilities.

    :param policy: policy
    :param obs: observation at the end of the expert's episode
    :param B_expert: the expert's DAG
    :return: loss
    """
    u, logits = policy.decoder_params(obs)
    order = tpolic.topological_order(B_expert, u.detach().numpy())
    orders = torch.as_tensor([order])
    B = torch.as_tensor(B_expert).unsqueeze(0)
    loss = -tpolic.order_edge_log_prob(u, logits, orders, B)[0]
    return loss


# #############################################################################
# Phase B: decoder RLOO
# #############################################################################


def rloo_loss(log_probs: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
    """
    Compute the REINFORCE leave-one-out surrogate loss of M samples.

    Its gradient is minus an unbiased estimate of the gradient of the expected
    reward.

    :param log_probs: log-probabilities of the samples, shape (M,)
    :param rewards: rewards of the samples, shape (M,)
    :return: loss
    """
    n_samples = rewards.shape[0]
    hdbg.dassert_lt(1, n_samples)
    baseline = (rewards.sum() - rewards) / (n_samples - 1)
    advantages = (rewards - baseline).detach()
    loss = -(advantages * log_probs).mean()
    return loss


def decoder_rloo_loss(
    policy: tpolic.TukeyPolicy,
    obs: tfeat.Observation,
    score_fn: Callable[[np.ndarray], float],
    *,
    n_samples: int,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample graphs at one state and compute the RLOO loss of the decoder.

    :param policy: policy
    :param obs: observation
    :param score_fn: exact score of a DAG (`TukeyEnv.reward_oracle()`)
    :param n_samples: graphs per state (M)
    :param generator: random generator
    :return: loss and the sampled rewards
    """
    u, logits = policy.decoder_params(obs)
    orders, B = tpolic.sample_graphs(u, logits, n_samples, generator)
    dtype = u.dtype
    rewards = torch.as_tensor(
        [score_fn(B[m].numpy()) for m in range(n_samples)], dtype=dtype
    )
    log_probs = tpolic.order_edge_log_prob(u, logits, orders, B)
    loss = rloo_loss(log_probs, rewards)
    return loss, rewards


# #############################################################################
# Phase C: tool REINFORCE
# #############################################################################


def estimate_phi(
    decoder_policy: tpolic.TukeyPolicy,
    obs: tfeat.Observation,
    score_fn: Callable[[np.ndarray], float],
    *,
    n_samples: int,
    generator: torch.Generator,
) -> Tuple[float, float]:
    """
    Estimate the frozen decoder's expected score at a state.

    :param decoder_policy: policy whose decoder is frozen
    :param obs: observation
    :param score_fn: exact score of a DAG
    :param n_samples: decoder samples (M)
    :param generator: random generator
    :return: mean score of M samples and the score of the first sample
    """
    with torch.no_grad():
        u, logits = decoder_policy.decoder_params(obs)
        _, B = tpolic.sample_graphs(u, logits, n_samples, generator)
    scores = [score_fn(B[m].numpy()) for m in range(n_samples)]
    return float(np.mean(scores)), float(scores[0])


@dataclasses.dataclass
class Rollout:
    """
    One tool-head episode, ready for `tool_reinforce_loss()`.
    """

    # log pi(a_t | s_t) of each decision (connected to the tool tower).
    log_probs: List[torch.Tensor]
    # Entropy of the action distribution at each decision.
    entropies: List[torch.Tensor]
    # Phi(s_t) at each decision state (shaping only).
    phi: List[float]
    # Phi(s_T): mean score of M decoder samples at the final state.
    phi_final: float
    # Score of the one graph emitted at the final state.
    sampled_reward: float
    # Tool calls made at or after each decision.
    calls_from: List[int]
    n_calls: int


def tool_rollout(
    policy: tpolic.TukeyPolicy,
    decoder_policy: tpolic.TukeyPolicy,
    env: tenv.TukeyEnv,
    *,
    seed: int,
    n_phi_samples: int,
    generator: torch.Generator,
    shaping: bool,
    legal_fn: Optional[LegalFn] = None,
) -> Rollout:
    """
    Play the tool head for one episode and record what the losses need.

    :param policy: policy whose tool tower acts (gradients flow)
    :param decoder_policy: policy whose decoder is frozen
    :param env: environment
    :param seed: instance seed
    :param n_phi_samples: decoder samples per Phi estimate (M)
    :param generator: random generator
    :param shaping: estimate Phi at every decision state
    :param legal_fn: legal-action mask
        - Default: None means `tukey_policy.legal_mask`
    :return: rollout record
    """
    legal_fn = legal_fn or tpolic.legal_mask
    evidence, _ = env.reset(seed=seed)
    score_fn = env.reward_oracle()
    actions, _ = policy.candidates(evidence.n_nodes)
    log_probs, entropies, phi, is_call = [], [], [], []
    while evidence.budget_left > 0:
        obs = tfeat.featurize(evidence)
        if shaping:
            phi_t, _ = estimate_phi(
                decoder_policy,
                obs,
                score_fn,
                n_samples=n_phi_samples,
                generator=generator,
            )
            phi.append(phi_t)
        all_log_probs = policy.tool_log_probs(obs, legal_fn(actions, evidence))
        probs = all_log_probs.detach().exp()
        choice = int(torch.multinomial(probs, 1, generator=generator))
        log_probs.append(all_log_probs[choice])
        finite = torch.isfinite(all_log_probs)
        entropies.append(
            -(all_log_probs[finite].exp() * all_log_probs[finite]).sum()
        )
        action = actions[choice]
        if action.kind == ttools.STOP:
            is_call.append(0)
            break
        is_call.append(1)
        evidence, _, _, _, _ = env.step(action)
    phi_final, sampled_reward = estimate_phi(
        decoder_policy,
        tfeat.featurize(evidence),
        score_fn,
        n_samples=n_phi_samples,
        generator=generator,
    )
    calls_from = list(np.cumsum(is_call[::-1])[::-1]) if is_call else []
    rollout = Rollout(
        log_probs=log_probs,
        entropies=entropies,
        phi=phi,
        phi_final=phi_final,
        sampled_reward=sampled_reward,
        calls_from=[int(c) for c in calls_from],
        n_calls=int(sum(is_call)),
    )
    return rollout


def tool_advantages(
    rollouts: List[Rollout], *, cost: float, shaping: bool
) -> List[List[float]]:
    """
    Compute the advantage of every decision of K rollouts on one instance.

    With shaping: `Phi(s_T) - b(s_t) - c * calls_from_t`, where `b(s_t)` is
    `rollout.phi[t]` (a sampled Phi or a learned value, never a function of
    the action at t). Without shaping: the rollout's total return minus the
    leave-one-out mean of the other rollouts' totals, for every decision.

    :param rollouts: K rollouts on one instance
    :param cost: training cost per tool call (c)
    :param shaping: use the potential-based return
    :return: advantages per rollout and decision
    """
    n_rollouts = len(rollouts)
    advantages = []
    if shaping:
        for rollout in rollouts:
            advantages.append(
                [
                    float(rollout.phi_final - rollout.phi[t] - cost * calls)
                    for t, calls in enumerate(rollout.calls_from)
                ]
            )
    else:
        hdbg.dassert_lt(1, n_rollouts)
        totals = np.array(
            [r.sampled_reward - cost * r.n_calls for r in rollouts]
        )
        for k, rollout in enumerate(rollouts):
            baseline = (totals.sum() - totals[k]) / (n_rollouts - 1)
            advantage = float(totals[k] - baseline)
            advantages.append([advantage] * len(rollout.calls_from))
    return advantages


def tool_reinforce_loss(
    rollouts: List[Rollout], *, cost: float, shaping: bool
) -> torch.Tensor:
    """
    Compute the tool head's REINFORCE surrogate loss for K rollouts.

    All rollouts must be on the same instance (the leave-one-out baseline of
    the unshaped estimator compares them).

    :param rollouts: K rollouts on one instance
    :param cost: training cost per tool call (c)
    :param shaping: use the potential-based return
    :return: loss (its gradient is minus an unbiased estimate of grad J)
    """
    advantages = tool_advantages(rollouts, cost=cost, shaping=shaping)
    terms = [
        -advantage * log_prob
        for rollout, rollout_adv in zip(rollouts, advantages)
        for log_prob, advantage in zip(rollout.log_probs, rollout_adv)
    ]
    if not terms:
        # Every rollout stopped with no decision (budget 0).
        return torch.zeros((), requires_grad=False)
    loss = torch.stack(terms).sum() / len(rollouts)
    return loss


# #############################################################################
# Batched training steps (gradient accumulation in chunks)
# #############################################################################

# States per forward pass when recomputing with gradients.
_CHUNK = 256


def _chunks(positions: List[int], size: int) -> List[List[int]]:
    """
    Split positions into consecutive chunks.

    :param positions: item positions
    :param size: chunk size
    :return: chunks
    """
    return [positions[k : k + size] for k in range(0, len(positions), size)]


def accumulate_bc_tool(
    policy: tpolic.TukeyPolicy,
    decisions: List[troll.Decision],
    *,
    weight: float = 1.0,
) -> Tuple[float, float]:
    """
    Backpropagate the mean negative log-likelihood of expert decisions.

    :param policy: policy
    :param decisions: expert decisions
    :param weight: loss weight
    :return: mean loss and next-action accuracy (argmax equals the expert)
    """
    n_total = len(decisions)
    hdbg.dassert_lt(0, n_total)
    total_nll = 0.0
    n_correct = 0
    groups = troll.group_by_nodes(decisions, lambda d: d.obs.node.shape[0])
    for positions in groups.values():
        for chunk in _chunks(positions, _CHUNK):
            batch = tpolic.stack_observations(
                [decisions[k].obs for k in chunk], dtype=policy.dtype
            )
            legal = np.stack([decisions[k].legal for k in chunk])
            log_probs, _ = policy.tool_log_probs_batch(batch, legal)
            target = torch.as_tensor([decisions[k].action_index for k in chunk])
            nll = -log_probs[torch.arange(len(chunk)), target]
            (weight * nll.sum() / n_total).backward()
            total_nll += float(nll.detach().sum())
            n_correct += int((log_probs.argmax(dim=-1) == target).sum())
    return total_nll / n_total, n_correct / n_total


def bc_accuracy(
    policy: tpolic.TukeyPolicy, decisions: List[troll.Decision]
) -> float:
    """
    Measure how often the policy's most likely action is the expert's.

    :param policy: policy
    :param decisions: expert decisions
    :return: fraction of decisions where the argmax equals the expert action
    """
    n_correct = 0
    groups = troll.group_by_nodes(decisions, lambda d: d.obs.node.shape[0])
    with torch.no_grad():
        for positions in groups.values():
            for chunk in _chunks(positions, _CHUNK):
                batch = tpolic.stack_observations(
                    [decisions[k].obs for k in chunk], dtype=policy.dtype
                )
                legal = np.stack([decisions[k].legal for k in chunk])
                log_probs, _ = policy.tool_log_probs_batch(batch, legal)
                target = torch.as_tensor(
                    [decisions[k].action_index for k in chunk]
                )
                n_correct += int((log_probs.argmax(dim=-1) == target).sum())
    return n_correct / max(1, len(decisions))


def accumulate_bc_decoder(
    policy: tpolic.TukeyPolicy,
    finals: List[Tuple[tfeat.Observation, np.ndarray]],
    *,
    weight: float = 1.0,
) -> float:
    """
    Backpropagate `-log p(G_expert, order)` averaged over final states.

    :param policy: policy
    :param finals: `(final observation, expert DAG)` pairs
    :param weight: loss weight
    :return: mean loss
    """
    n_total = len(finals)
    total = 0.0
    groups = troll.group_by_nodes(finals, lambda f: f[0].node.shape[0])
    for positions in groups.values():
        for chunk in _chunks(positions, _CHUNK):
            batch = tpolic.stack_observations(
                [finals[k][0] for k in chunk], dtype=policy.dtype
            )
            u, logits = policy.decoder_params_batch(batch)
            u_np = u.detach().numpy()
            orders = torch.as_tensor(
                [
                    [tpolic.topological_order(finals[k][1], u_np[row])]
                    for row, k in enumerate(chunk)
                ]
            )
            B = torch.as_tensor(np.stack([finals[k][1] for k in chunk]))[:, None]
            nll = -tpolic.order_edge_log_prob(u, logits, orders, B)[:, 0]
            (weight * nll.sum() / n_total).backward()
            total += float(nll.detach().sum())
    return total / n_total


def accumulate_decoder_rloo(
    policy: tpolic.TukeyPolicy,
    states: List[Tuple[tfeat.Observation, Callable[[np.ndarray], float]]],
    *,
    n_samples: int,
    generator: torch.Generator,
    weight: float = 1.0,
) -> float:
    """
    Backpropagate the decoder's RLOO loss averaged over states.

    :param policy: policy
    :param states: `(observation, exact score function)` pairs
    :param n_samples: graphs per state (M)
    :param generator: random generator
    :param weight: loss weight
    :return: mean score of the sampled graphs
    """
    n_total = len(states)
    total_reward = 0.0
    groups = troll.group_by_nodes(states, lambda s: s[0].node.shape[0])
    for positions in groups.values():
        for chunk in _chunks(positions, _CHUNK):
            batch = tpolic.stack_observations(
                [states[k][0] for k in chunk], dtype=policy.dtype
            )
            u, logits = policy.decoder_params_batch(batch)
            orders, B = tpolic.sample_graphs(u, logits, n_samples, generator)
            rewards = torch.as_tensor(
                [
                    [states[k][1](B[row, m].numpy()) for m in range(n_samples)]
                    for row, k in enumerate(chunk)
                ],
                dtype=u.dtype,
            )
            log_probs = tpolic.order_edge_log_prob(u, logits, orders, B)
            baseline = (rewards.sum(dim=1, keepdim=True) - rewards) / (
                n_samples - 1
            )
            advantages = (rewards - baseline).detach()
            loss = -(advantages * log_probs).mean(dim=1).sum() / n_total
            (weight * loss).backward()
            total_reward += float(rewards.mean(dim=1).sum())
    return total_reward / n_total


def terminal_phi(
    decoder_policy: tpolic.TukeyPolicy,
    episodes: List[troll.Episode],
    *,
    n_samples: int,
    generator: torch.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate Phi at each episode's final state with M decoder samples.

    :param decoder_policy: policy whose decoder is frozen
    :param episodes: episodes
    :param n_samples: decoder samples (M)
    :param generator: random generator
    :return: mean sampled score and the first sample's score, per episode
    """
    phi = np.zeros(len(episodes))
    first = np.zeros(len(episodes))
    groups = troll.group_by_nodes(episodes, lambda e: e.task.cfg.n_nodes)
    for positions in groups.values():
        for chunk in _chunks(positions, _CHUNK):
            batch = tpolic.stack_observations(
                [episodes[k].final_obs for k in chunk],
                dtype=decoder_policy.dtype,
            )
            with torch.no_grad():
                u, logits = decoder_policy.decoder_params_batch(batch)
            _, B = tpolic.sample_graphs(u, logits, n_samples, generator)
            for row, k in enumerate(chunk):
                scores = [
                    episodes[k].score_fn(B[row, m].numpy())
                    for m in range(n_samples)
                ]
                phi[k] = np.mean(scores)
                first[k] = scores[0]
    return phi, first


def accumulate_tool_rl(
    policy: tpolic.TukeyPolicy,
    decoder_policy: tpolic.TukeyPolicy,
    groups: List[List[troll.Episode]],
    *,
    cost: float,
    shaping: bool,
    n_phi_samples: int,
    generator: torch.Generator,
    entropy_coef: float = 0.0,
    value_coef: float = 1.0,
) -> Dict[str, float]:
    """
    Backpropagate the tool head's REINFORCE loss for K rollouts per instance.

    Rollouts were played without gradients; log-probabilities are recomputed
    here in chunks. With shaping the baseline of each decision is the value
    head's prediction at play time (it does not depend on the action), and the
    terminal value is the mean of M decoder samples. Advantages come from
    `tool_advantages()`, the function the gradient checks verify.

    :param policy: policy whose tool tower and value head learn
    :param decoder_policy: policy whose decoder is frozen
    :param groups: K episodes per instance
    :param cost: training cost per tool call (c)
    :param shaping: use the potential-based return
    :param n_phi_samples: decoder samples per terminal estimate (M)
    :param generator: random generator
    :param entropy_coef: weight of the entropy bonus
    :param value_coef: weight of the value-head regression
    :return: statistics of the batch
    """
    episodes = [episode for group in groups for episode in group]
    phi_final, first_score = terminal_phi(
        decoder_policy, episodes, n_samples=n_phi_samples, generator=generator
    )
    # Advantages and value targets for every decision.
    items = []
    position = 0
    for group in groups:
        rollouts = []
        for episode in group:
            rollouts.append(
                Rollout(
                    log_probs=[],
                    entropies=[],
                    phi=[d.value for d in episode.decisions],
                    phi_final=float(phi_final[position]),
                    sampled_reward=float(first_score[position]),
                    calls_from=episode.calls_from,
                    n_calls=episode.n_calls,
                )
            )
            position += 1
        advantages = tool_advantages(rollouts, cost=cost, shaping=shaping)
        for episode, rollout, adv in zip(group, rollouts, advantages):
            for t, decision in enumerate(episode.decisions):
                target = rollout.phi_final - cost * rollout.calls_from[t]
                items.append((decision, adv[t], target, len(group)))
    n_groups = len(groups)
    n_items = max(1, len(items))
    stats = {"policy_loss": 0.0, "entropy": 0.0, "value_loss": 0.0}
    by_nodes = troll.group_by_nodes(items, lambda it: it[0].obs.node.shape[0])
    for positions in by_nodes.values():
        for chunk in _chunks(positions, _CHUNK):
            batch = tpolic.stack_observations(
                [items[k][0].obs for k in chunk], dtype=policy.dtype
            )
            legal = np.stack([items[k][0].legal for k in chunk])
            log_probs, values = policy.tool_log_probs_batch(batch, legal)
            target_idx = torch.as_tensor(
                [items[k][0].action_index for k in chunk]
            )
            chosen = log_probs[torch.arange(len(chunk)), target_idx]
            dtype = log_probs.dtype
            adv = torch.as_tensor([items[k][1] for k in chunk], dtype=dtype)
            k_rollouts = torch.as_tensor(
                [items[k][3] for k in chunk], dtype=dtype
            )
            policy_loss = -(adv * chosen / k_rollouts).sum() / n_groups
            probs = log_probs.exp()
            entropy = -(probs * log_probs.masked_fill(probs == 0, 0.0)).sum(
                dim=-1
            )
            targets = torch.as_tensor([items[k][2] for k in chunk], dtype=dtype)
            value_loss = ((values - targets) ** 2).sum() / n_items
            loss = (
                policy_loss
                - entropy_coef * entropy.sum() / n_items
                + value_coef * value_loss
            )
            loss.backward()
            stats["policy_loss"] += policy_loss.item()
            stats["entropy"] += entropy.detach().sum().item() / n_items
            stats["value_loss"] += value_loss.item()
    stats["phi_final_mean"] = float(np.mean(phi_final))
    stats["calls_mean"] = float(np.mean([e.n_calls for e in episodes]))
    return stats
