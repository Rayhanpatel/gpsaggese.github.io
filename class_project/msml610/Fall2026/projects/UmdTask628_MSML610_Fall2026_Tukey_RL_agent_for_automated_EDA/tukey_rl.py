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
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch

import helpers.hdbg as hdbg
import tukey_env as tenv
import tukey_features as tfeat
import tukey_policy as tpolic
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
    n_rollouts = len(rollouts)
    terms = []
    if shaping:
        for rollout in rollouts:
            for t, log_prob in enumerate(rollout.log_probs):
                advantage = float(
                    rollout.phi_final
                    - rollout.phi[t]
                    - cost * rollout.calls_from[t]
                )
                terms.append(-advantage * log_prob)
    else:
        hdbg.dassert_lt(1, n_rollouts)
        totals = np.array(
            [r.sampled_reward - cost * r.n_calls for r in rollouts]
        )
        for k, rollout in enumerate(rollouts):
            baseline = (totals.sum() - totals[k]) / (n_rollouts - 1)
            advantage = float(totals[k] - baseline)
            for log_prob in rollout.log_probs:
                terms.append(-advantage * log_prob)
    if not terms:
        # Every rollout stopped with no decision (budget 0).
        return torch.zeros((), requires_grad=False)
    loss = torch.stack(terms).sum() / n_rollouts
    return loss
