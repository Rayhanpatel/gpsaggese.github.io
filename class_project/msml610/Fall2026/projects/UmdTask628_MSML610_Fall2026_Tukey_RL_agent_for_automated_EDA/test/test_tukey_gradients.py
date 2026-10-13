"""
Gate G4: gradient checks for the policy and the training estimators.

- (a) Autograd against central finite differences in float64 on the
  log-probabilities of the masked tool softmax, the Plackett-Luce order and the
  edge Bernoullis: relative error below 1e-4
- (b) Unbiasedness on an enumerable toy (3 variables, at most two calls among
  three tools, plus STOP): the mean of 20,000 estimates from each training
  loss matches the exact gradient, computed by enumerating every trajectory
  and every decoder outcome, within 3 standard errors; checked for decoder
  RLOO and for tool REINFORCE with and without shaping
- (c) Sign: on a two-armed bandit, gradient steps on each loss raise the
  probability of the better arm

Import as:

import test.test_tukey_gradients as ttetugra
"""

import itertools
import logging
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch

import helpers.hunit_test as hunitest
import tukey_env as tenv
import tukey_features as tfeat
import tukey_policy as tpolic
import tukey_rl as trl
import tukey_tools as ttools

_LOG = logging.getLogger(__name__)

_N_DRAWS = 20_000


# #############################################################################
# Helpers
# #############################################################################


def _tool_params(policy: tpolic.TukeyPolicy) -> List[torch.nn.Parameter]:
    """
    Return the tool tower's parameters.

    :param policy: policy
    :return: parameters of the tool encoder and head
    """
    params = list(policy.tool_encoder.parameters())
    params += list(policy.tool_head.parameters())
    return params


def _decoder_params(policy: tpolic.TukeyPolicy) -> List[torch.nn.Parameter]:
    """
    Return the decoder tower's parameters.

    :param policy: policy
    :return: parameters of the decoder encoder and scores
    """
    params = list(policy.decoder_encoder.parameters())
    params += list(policy.node_score.parameters())
    params += list(policy.edge_score.parameters())
    return params


def _direction(
    params: List[torch.nn.Parameter], seed: int
) -> List[torch.Tensor]:
    """
    Draw a random unit direction in parameter space.

    :param params: parameters
    :param seed: random seed
    :return: one tensor per parameter, jointly of norm 1
    """
    generator = torch.Generator().manual_seed(seed)
    direction = [
        torch.randn(p.shape, generator=generator, dtype=p.dtype) for p in params
    ]
    norm = torch.sqrt(sum((v**2).sum() for v in direction))
    return [v / norm for v in direction]


def _autograd_slope(
    value: torch.Tensor,
    params: List[torch.nn.Parameter],
    direction: List[torch.Tensor],
) -> float:
    """
    Compute the directional derivative of a scalar by autograd.

    :param value: scalar connected to `params`
    :param params: parameters
    :param direction: direction
    :return: gradient dotted with the direction
    """
    grads = torch.autograd.grad(
        value, params, allow_unused=True, retain_graph=True
    )
    slope = sum(
        float((g * v).sum()) for g, v in zip(grads, direction) if g is not None
    )
    return slope


def _finite_difference_slope(
    func: Callable[[], torch.Tensor],
    params: List[torch.nn.Parameter],
    direction: List[torch.Tensor],
    *,
    eps: float = 1e-6,
) -> float:
    """
    Compute a central finite difference along a direction.

    :param func: function of the parameters returning a scalar
    :param params: parameters
    :param direction: direction
    :param eps: step
    :return: `(f(p + eps v) - f(p - eps v)) / (2 eps)`
    """
    with torch.no_grad():
        for p, v in zip(params, direction):
            p.add_(eps * v)
        f_plus = float(func())
        for p, v in zip(params, direction):
            p.sub_(2 * eps * v)
        f_minus = float(func())
        for p, v in zip(params, direction):
            p.add_(eps * v)
    return (f_plus - f_minus) / (2 * eps)


def _all_outcomes(n_nodes: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    List every (order, edges-along-the-order) outcome of the decoder.

    :param n_nodes: number of variables
    :return: orders (N, d) and graphs (N, d, d)
    """
    orders = []
    graphs = []
    n_pairs = n_nodes * (n_nodes - 1) // 2
    for order in itertools.permutations(range(n_nodes)):
        pairs = [
            (order[a], order[b])
            for a, b in itertools.combinations(range(n_nodes), 2)
        ]
        for bits in itertools.product([0, 1], repeat=n_pairs):
            B = np.zeros((n_nodes, n_nodes), dtype=int)
            for (i, j), bit in zip(pairs, bits):
                B[i, j] = bit
            orders.append(list(order))
            graphs.append(B)
    return torch.as_tensor(orders), torch.as_tensor(np.array(graphs))


def _check_unbiased(
    test: hunitest.TestCase, estimates: np.ndarray, exact: float, name: str
) -> str:
    """
    Assert that a sample mean is within 3 standard errors of the exact value.

    :param test: calling test case
    :param estimates: estimates of a directional derivative
    :param exact: exact directional derivative
    :param name: estimator name for the log
    :return: summary line
    """
    mean = float(estimates.mean())
    std_err = float(estimates.std(ddof=1) / np.sqrt(len(estimates)))
    z_score = (mean - exact) / std_err
    summary = (
        f"{name}: exact {exact:.6f}, mean {mean:.6f}, se {std_err:.6f}, "
        f"z {z_score:.2f}, sd {estimates.std(ddof=1):.5f}"
    )
    _LOG.info(summary)
    test.assertLess(abs(z_score), 3.0, msg=summary)
    return summary


# #############################################################################
# Test_finite_differences
# #############################################################################


class Test_finite_differences(hunitest.TestCase):
    """
    (a) Autograd slopes equal central finite differences in float64.
    """

    def _setup(
        self,
    ) -> Tuple[tpolic.TukeyPolicy, tfeat.Observation, np.ndarray]:
        """
        Build a float64 policy and a state with some evidence.

        :return: policy, observation and legal mask
        """
        torch.manual_seed(10)
        policy = tpolic.TukeyPolicy(hidden=16).double()
        cfg = tenv.GraphConfig(family="mixed", n_nodes=4, n_edges=4)
        env = tenv.TukeyEnv(cfg, budget=20)
        evidence, _ = env.reset(seed=10)
        actions = ttools.enumerate_tool_actions(4)
        for k in (0, 5, 9, 20, 33, 41):
            evidence, _, _, _, _ = env.step(actions[k])
        all_actions, _ = policy.candidates(4)
        legal = tpolic.legal_mask(all_actions, evidence)
        return policy, tfeat.featurize(evidence), legal

    def _check(
        self,
        func: Callable[[], torch.Tensor],
        params: List[torch.nn.Parameter],
    ) -> float:
        """
        Compare autograd and finite differences along 6 unit directions.

        The directions are the gradient itself and 5 random ones. The error is
        relative to the gradient norm: a random direction in many dimensions
        is nearly orthogonal to the gradient, and its tiny slope is below the
        roundoff of the finite difference (a defect of the reference, not of
        the gradient).

        :param func: scalar function of the parameters
        :param params: parameters to perturb
        :return: largest `|slope_autograd - slope_fd| / |grad|`
        """
        grads = torch.autograd.grad(func(), params, allow_unused=True)
        grads = [
            torch.zeros_like(p) if g is None else g
            for g, p in zip(grads, params)
        ]
        grad_norm = float(torch.sqrt(sum((g**2).sum() for g in grads)))
        directions = [[g / grad_norm for g in grads]]
        directions += [_direction(params, seed) for seed in range(5)]
        worst = 0.0
        for direction in directions:
            slope_ad = _autograd_slope(func(), params, direction)
            slope_fd = _finite_difference_slope(
                func, params, direction, eps=1e-5
            )
            worst = max(worst, abs(slope_ad - slope_fd) / grad_norm)
        return worst

    def test1(self) -> None:
        """
        Masked tool softmax: weighted sum of legal log-probabilities.
        """
        policy, obs, legal = self._setup()
        weights = torch.as_tensor(
            np.random.default_rng(0).normal(size=int(legal.sum()))
        )
        legal_t = torch.as_tensor(legal)

        def _func() -> torch.Tensor:
            return (policy.tool_log_probs(obs, legal)[legal_t] * weights).sum()

        self.assertLess(self._check(_func, _tool_params(policy)), 1e-4)

    def test2(self) -> None:
        """
        Plackett-Luce order and edge Bernoullis, separately and jointly.
        """
        policy, obs, _ = self._setup()
        u, logits = policy.decoder_params(obs)
        orders, B = tpolic.sample_graphs(
            u, logits, 16, torch.Generator().manual_seed(0)
        )

        def _func() -> torch.Tensor:
            u, logits = policy.decoder_params(obs)
            return tpolic.order_edge_log_prob(u, logits, orders, B).sum()

        node_params = list(policy.node_score.parameters())
        edge_params = list(policy.edge_score.parameters())
        lines = [
            f"order {self._check(_func, node_params) < 1e-4}",
            f"edges {self._check(_func, edge_params) < 1e-4}",
            f"joint {self._check(_func, _decoder_params(policy)) < 1e-4}",
        ]
        self.assert_equal("\n".join(lines), "order True\nedges True\njoint True")


# #############################################################################
# Test_unbiased_estimators
# #############################################################################


class _Toy:
    """
    Enumerable toy: 3 variables, budget 2, three allowed tools plus STOP.
    """

    def _make_decoder_informative(self, B_true: np.ndarray) -> None:
        """
        Train the decoder so its expected score depends on the state.

        At random initialization the decoder's expected score is nearly the
        same in every state, so a check of tool-gradient estimators would have
        no power. Behaviour cloning on state-dependent targets (G* once
        `lingam_dir` was called, the empty graph otherwise) fixes that.

        :param B_true: true DAG of the toy instance
        """
        lingam = self.tools[1]
        empty = np.zeros_like(B_true)
        params = _decoder_params(self.policy)
        # 60 small steps: informative but still stochastic (a deterministic
        # decoder would make every RLOO estimate zero).
        optimizer = torch.optim.Adam(params, lr=0.003)
        for _ in range(60):
            loss = sum(
                trl.bc_decoder_loss(
                    self.policy,
                    self.obs[p][0],
                    B_true if lingam in p else empty,
                )
                for p in self.prefixes
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def _observe(
        self, prefix: Tuple[ttools.Action, ...]
    ) -> Tuple[tfeat.Observation, np.ndarray]:
        """
        Replay a prefix of tool calls.
        """
        evidence, _ = self.env.reset(seed=self.seed)
        for action in prefix:
            evidence, _, _, _, _ = self.env.step(action)
        return tfeat.featurize(evidence), self.legal_fn(self.actions, evidence)

    def __init__(self) -> None:
        torch.manual_seed(20)
        self.policy = tpolic.TukeyPolicy(hidden=16).double()
        cfg = tenv.GraphConfig(family="linear_exp", n_nodes=3, n_edges=2)
        self.env = tenv.TukeyEnv(cfg, budget=2)
        self.seed = 20
        self.tools = [
            ttools.ci_test_action(0, 1, (2,)),
            ttools.Action(ttools.LINGAM_DIR, (1, 2)),
            ttools.Action(ttools.NONGAUSS, (0,)),
        ]
        self.stop = ttools.Action(ttools.STOP)
        self.cost = 0.05
        self.actions, _ = self.policy.candidates(3)
        self.position = {a: k for k, a in enumerate(self.actions)}
        self.orders, self.graphs = _all_outcomes(3)
        self.prefixes = [()] + [(t,) for t in self.tools]
        self.prefixes += [p for p in itertools.permutations(self.tools, 2)]
        self.env.reset(seed=self.seed)
        score_fn = self.env.reward_oracle()
        self.scores = np.array([score_fn(B.numpy()) for B in self.graphs])
        self.obs = {p: self._observe(p) for p in self.prefixes}
        self._make_decoder_informative(self.env._instance.truth.B)

    def legal_fn(
        self, actions: List[ttools.Action], evidence: ttools.Evidence
    ) -> np.ndarray:
        """
        Allow the three toy tools (unused, within budget) and STOP.
        """
        mask = [
            a == self.stop
            or (
                a in self.tools
                and a not in evidence.results
                and evidence.budget_left > 0
            )
            for a in actions
        ]
        return np.array(mask)

    def trajectories(
        self,
    ) -> List[List[Tuple[Tuple[ttools.Action, ...], ttools.Action]]]:
        """
        List every trajectory as its decisions `(prefix, action)`.
        """
        trajs = [[((), self.stop)]]
        for t in self.tools:
            trajs.append([((), t), ((t,), self.stop)])
        for t1, t2 in itertools.permutations(self.tools, 2):
            trajs.append([((), t1), ((t1,), t2)])
        return trajs

    @staticmethod
    def final_prefix(traj) -> Tuple[ttools.Action, ...]:
        """
        Return the state where a trajectory ends.
        """
        prefix, action = traj[-1]
        return prefix if action.kind == ttools.STOP else prefix + (action,)

    def tool_log_prob(self, prefix, action) -> torch.Tensor:
        """
        Compute log pi(action | prefix).
        """
        obs, legal = self.obs[prefix]
        return self.policy.tool_log_probs(obs, legal)[self.position[action]]

    def decoder_log_probs(self, prefix) -> torch.Tensor:
        """
        Compute the log-probability of all 48 decoder outcomes at a state.
        """
        u, logits = self.policy.decoder_params(self.obs[prefix][0])
        return tpolic.order_edge_log_prob(u, logits, self.orders, self.graphs)


class Test_unbiased_estimators(hunitest.TestCase):
    """
    (b) Training estimators are unbiased on an enumerable toy.
    """

    def test1(self) -> None:
        """
        Decoder RLOO, tool REINFORCE without and with shaping.
        """
        toy = _Toy()
        trajs = toy.trajectories()
        self.assertEqual(len(trajs), 10)
        tool_params = _tool_params(toy.policy)
        dec_params = _decoder_params(toy.policy)
        v_tool = _direction(tool_params, 1)
        v_dec = _direction(dec_params, 2)
        scores_t = torch.as_tensor(toy.scores)
        # Exact objective by enumeration.
        traj_log_probs = [
            sum(toy.tool_log_prob(p, a) for p, a in traj) for traj in trajs
        ]
        expected_score = {
            p: (toy.decoder_log_probs(p).exp() * scores_t).sum()
            for p in toy.prefixes
        }
        # The toy is informative: Phi differs between states.
        phis = np.array([float(v) for v in expected_score.values()])
        _LOG.info("Phi per state: %s", np.round(phis, 3).tolist())
        self.assertGreater(phis.max() - phis.min(), 0.3)
        s_star = (toy.tools[0], toy.tools[1])
        max_prob = float(toy.decoder_log_probs(s_star).exp().max())
        self.assertLess(max_prob, 0.9)
        n_calls = [sum(a.kind != ttools.STOP for _, a in traj) for traj in trajs]
        J = sum(
            lp.exp()
            * (expected_score[toy.final_prefix(traj)] - toy.cost * calls)
            for lp, traj, calls in zip(traj_log_probs, trajs, n_calls)
        )
        exact_tool = _autograd_slope(J, tool_params, v_tool)
        exact_dec = _autograd_slope(expected_score[s_star], dec_params, v_dec)
        # Slopes of each decision's log-probability along v_tool.
        decision_slope: Dict[Tuple, float] = {}
        for traj in trajs:
            for prefix, action in traj:
                if (prefix, action) not in decision_slope:
                    lp = toy.tool_log_prob(prefix, action)
                    decision_slope[(prefix, action)] = _autograd_slope(
                        lp, tool_params, v_tool
                    )
        self.assertEqual(len(decision_slope), 13)
        dec_log_probs = toy.decoder_log_probs(s_star)
        dec_slopes = np.array(
            [
                _autograd_slope(dec_log_probs[k], dec_params, v_dec)
                for k in range(len(toy.scores))
            ]
        )
        traj_probs = np.array([float(lp.exp()) for lp in traj_log_probs])
        outcome_probs = {
            p: toy.decoder_log_probs(p).exp().detach().numpy()
            for p in toy.prefixes
        }
        rng = np.random.default_rng(0)

        def _sample_scores(prefix, size: int) -> np.ndarray:
            probs = outcome_probs[prefix] / outcome_probs[prefix].sum()
            return toy.scores[rng.choice(len(probs), size=size, p=probs)]

        def _slope_of(loss: torch.Tensor, eps: torch.Tensor) -> float:
            return -float(torch.autograd.grad(loss, eps, retain_graph=True)[0])

        estimates: Dict[str, List[float]] = {
            "decoder_rloo": [],
            "tool_unshaped": [],
            "tool_shaped": [],
            "biased_control": [],
        }

        def _next_prefix(prefix, action):
            return prefix if action.kind == ttools.STOP else prefix + (action,)

        probs_star = outcome_probs[s_star] / outcome_probs[s_star].sum()
        for _ in range(_N_DRAWS):
            eps = torch.zeros((), dtype=torch.float64, requires_grad=True)
            # Decoder RLOO with M = 4 graphs at s*.
            picks = rng.choice(len(probs_star), size=4, p=probs_star)
            loss = trl.rloo_loss(
                eps * torch.as_tensor(dec_slopes[picks]),
                torch.as_tensor(toy.scores[picks]),
            )
            estimates["decoder_rloo"].append(_slope_of(loss, eps))
            # Tool REINFORCE with K = 4 rollouts on the instance.
            for shaping in (False, True):
                rollouts = []
                picked = rng.choice(
                    len(trajs), size=4, p=traj_probs / traj_probs.sum()
                )
                for k in picked:
                    traj = trajs[k]
                    final = toy.final_prefix(traj)
                    is_call = [int(a.kind != ttools.STOP) for _, a in traj]
                    rollouts.append(
                        trl.Rollout(
                            log_probs=[
                                eps * decision_slope[(p, a)] for p, a in traj
                            ],
                            entropies=[],
                            phi=[
                                float(_sample_scores(p, 4).mean())
                                for p, _ in traj
                            ],
                            phi_final=float(_sample_scores(final, 4).mean()),
                            sampled_reward=float(_sample_scores(final, 1)[0]),
                            calls_from=list(np.cumsum(is_call[::-1])[::-1]),
                            n_calls=sum(is_call),
                        )
                    )
                loss = trl.tool_reinforce_loss(
                    rollouts, cost=toy.cost, shaping=shaping
                )
                name = "tool_shaped" if shaping else "tool_unshaped"
                estimates[name].append(_slope_of(loss, eps))
                if shaping:
                    # Negative control: a "baseline" taken after the action
                    # depends on it, so the estimator is biased.
                    for rollout, k in zip(rollouts, picked):
                        rollout.phi = [
                            float(_sample_scores(_next_prefix(p, a), 4).mean())
                            for p, a in trajs[k]
                        ]
                    loss = trl.tool_reinforce_loss(
                        rollouts, cost=toy.cost, shaping=True
                    )
                    estimates["biased_control"].append(_slope_of(loss, eps))
        _check_unbiased(
            self, np.array(estimates["decoder_rloo"]), exact_dec, "decoder_rloo"
        )
        _check_unbiased(
            self,
            np.array(estimates["tool_unshaped"]),
            exact_tool,
            "tool_unshaped",
        )
        _check_unbiased(
            self, np.array(estimates["tool_shaped"]), exact_tool, "tool_shaped"
        )
        # The check has power: the biased control must fail it.
        with self.assertRaises(AssertionError):
            _check_unbiased(
                self,
                np.array(estimates["biased_control"]),
                exact_tool,
                "biased_control",
            )


# #############################################################################
# Test_bandit_sign
# #############################################################################


class Test_bandit_sign(hunitest.TestCase):
    """
    (c) Minimizing each loss raises the probability of the better arm.
    """

    def _train(self, make_loss: Callable) -> float:
        """
        Run 60 SGD steps on a two-armed bandit with rewards (1, 0).

        :param make_loss: function (log_probs, arms, rewards) -> loss
        :return: final probability of arm 0
        """
        theta = torch.zeros(2, dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.SGD([theta], lr=0.5)
        rewards = torch.as_tensor([1.0, 0.0], dtype=torch.float64)
        generator = torch.Generator().manual_seed(0)
        for _ in range(60):
            log_probs = torch.log_softmax(theta, dim=0)
            arms = torch.multinomial(
                log_probs.exp().detach(),
                8,
                replacement=True,
                generator=generator,
            )
            loss = make_loss(log_probs, arms, rewards)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return float(torch.softmax(theta, dim=0)[0])

    def test1(self) -> None:
        """
        RLOO and tool REINFORCE with and without shaping all learn arm 0.
        """

        def _rloo(log_probs, arms, rewards):
            return trl.rloo_loss(log_probs[arms], rewards[arms])

        def _tool(shaping: bool) -> Callable:
            def _loss(log_probs, arms, rewards):
                expected = float((log_probs.exp() * rewards).sum())
                rollouts = [
                    trl.Rollout(
                        log_probs=[log_probs[a]],
                        entropies=[],
                        phi=[expected],
                        phi_final=float(rewards[a]),
                        sampled_reward=float(rewards[a]),
                        calls_from=[0],
                        n_calls=0,
                    )
                    for a in arms
                ]
                return trl.tool_reinforce_loss(
                    rollouts, cost=0.0, shaping=shaping
                )

            return _loss

        final = [
            self._train(_rloo),
            self._train(_tool(False)),
            self._train(_tool(True)),
        ]
        self.assertTrue(all(p > 0.9 for p in final), msg=str(final))
