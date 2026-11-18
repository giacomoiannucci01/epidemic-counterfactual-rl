"""Ensemble DDQN with particle-filtered training and Monte Carlo planning."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import timedelta
from itertools import product
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .engine import Engine, START, CONTROL_END, END, AGES, DECISION_LENGTHS

ACTIONS = np.asarray(list(product((1., .7, .4, .2), repeat=3)))


@dataclass
class RunConfig:
    profile: str = "smoke"
    seed: int = 20261118
    deployment_particles: int = 32
    planning_particles: int = 16
    train_episodes: int = 2
    outer_iterations: int = 1
    ensemble_size: int = 1
    hidden_units: int = 48
    mc_candidates: int = 2
    mc_rollouts: int = 1
    batch_size: int = 8
    learning_rate: float = .0003
    target_sync_updates: int = 8
    alpha: float = .5
    lambda0: float | None = None
    eta0: float = .4
    oracle: bool = True
    planning_trace_limit: int = 3000

    @classmethod
    def research(cls):
        return cls(profile="research", deployment_particles=5000, planning_particles=1024,
                   train_episodes=2000, outer_iterations=20, ensemble_size=3, hidden_units=128,
                   mc_candidates=64, mc_rollouts=192, batch_size=512,
                   target_sync_updates=32, planning_trace_limit=10000)

    def validate(self):
        for key in ("deployment_particles", "planning_particles", "train_episodes", "outer_iterations",
                    "ensemble_size", "mc_candidates", "mc_rollouts", "batch_size"):
            if getattr(self, key) < 1:
                raise ValueError(f"{key} must be positive")
        if self.mc_candidates > len(ACTIONS) or not 0 <= self.alpha <= 1:
            raise ValueError("Invalid action candidates or alpha")


class QNetwork(nn.Module):
    def __init__(self, features, hidden):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(features, hidden), nn.SiLU(),
                                     nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, len(ACTIONS)))

    def forward(self, x):
        return self.network(x)


class EnsembleDDQN:
    def __init__(self, feature_count, config):
        self.config = config
        self.online, self.targets, self.optimizers = [], [], []
        self.replay, self.updates = [], 0
        for j in range(config.ensemble_size):
            torch.manual_seed(config.seed+j*1009)
            online = QNetwork(feature_count, config.hidden_units)
            target = QNetwork(feature_count, config.hidden_units)
            target.load_state_dict(online.state_dict())
            target.eval()
            self.online.append(online); self.targets.append(target)
            self.optimizers.append(torch.optim.AdamW(online.parameters(), lr=config.learning_rate, weight_decay=1e-5))

    def values(self, features):
        with torch.no_grad():
            x = torch.as_tensor(features, dtype=torch.float32)
            return torch.stack([net(x) for net in self.online]).mean(0).numpy()

    def choose(self, features, rng, epsilon=0.):
        return int(rng.integers(len(ACTIONS))) if rng.random() < epsilon else int(np.argmax(self.values(features)))

    def remember(self, state, action, health_reward, successor, terminal, price_coefficient):
        # Keep health and social components separate: old experiences must be
        # evaluated at the CURRENT dual price after an outer iteration changes it.
        self.replay.append((state.copy(), int(action), float(health_reward),
                            successor.copy(), bool(terminal), float(price_coefficient)))
        if len(self.replay) > 200000:
            del self.replay[:10000]

    @staticmethod
    def replay_rewards(batch, lam):
        return np.array([row[2] + lam*row[5] for row in batch], dtype=np.float32)

    def learn(self, rng, lam):
        if len(self.replay) < self.config.batch_size:
            return
        ids = rng.choice(len(self.replay), self.config.batch_size, replace=False)
        batch = [self.replay[i] for i in ids]
        states = torch.as_tensor(np.stack([r[0] for r in batch]), dtype=torch.float32)
        actions = torch.as_tensor([r[1] for r in batch], dtype=torch.long)
        reward = torch.as_tensor(self.replay_rewards(batch, lam), dtype=torch.float32)
        successor = torch.as_tensor(np.stack([r[3] for r in batch]), dtype=torch.float32)
        alive = torch.as_tensor([not r[4] for r in batch], dtype=torch.float32)
        for net, target, optimizer in zip(self.online, self.targets, self.optimizers):
            prediction = net(states).gather(1, actions[:, None]).squeeze(1)
            with torch.no_grad():
                next_actions = net(successor).argmax(1)
                # Double-DQN action selection/evaluation; undiscounted finite horizon.
                expected = reward + alive*target(successor).gather(1, next_actions[:, None]).squeeze(1)
            loss = nn.functional.smooth_l1_loss(prediction, expected)
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 10.)
            optimizer.step()
        self.updates += 1
        if self.updates % self.config.target_sync_updates == 0:
            for target, net in zip(self.targets, self.online):
                target.load_state_dict(net.state_dict())

    def save(self, path, feature_names):
        torch.save({"state_dicts": [n.state_dict() for n in self.online],
                    "feature_names": feature_names, "config": asdict(self.config),
                    "architecture": "ensemble Double DQN, two SiLU hidden layers"}, path)


class Experiment:
    def __init__(self, country, config, output):
        config.validate()
        torch.set_num_threads(1)
        self.config, self.output = config, Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.engine = Engine(country)
        self.initial = self.engine.initial_belief(config.deployment_particles, config.seed)
        self.agent = EnsembleDDQN(len(self.engine.feature_names()), config)
        self.planning_trace, self._trace_seen = [], 0
        self.decision_trace = []
        self.references = None

    def sample_belief(self, b, rng):
        idx = rng.choice(b.n, size=self.config.planning_particles, replace=True, p=b.weights)
        return b.subset(idx)

    def pending_deaths(self, truth, dt):
        m = self.engine.model(dt)
        community, ward, icu = self.engine.mortality
        hd = (1-m.p_icu)*ward + m.p_icu*icu
        ifr = (1-m.p_hospital)*community + m.p_hospital*hd
        x = truth.x[0]
        return float(((x[:, 1:5].sum(axis=1))*ifr + x[:, 5]*hd + x[:, 6]*icu).sum())

    def health_reward(self, deaths, icu, social, lam):
        dr, ur = self.references
        return -96*(self.config.alpha*deaths/dr + (1-self.config.alpha)*icu/ur + lam*social/self.engine.budget)

    def record_state(self, belief, dt, phase, triggers, diagnostic, extra=None):
        row = {"date": str(dt), "phase": phase, "oracle_reset": bool(triggers),
               "oracle_triggers": ";".join(triggers), **diagnostic, **self.engine.state_row(belief)}
        if extra:
            row.update(extra)
        if phase == "planning":
            self._trace_seen += 1
            limit = self.config.planning_trace_limit
            if limit < 0 or len(self.planning_trace) < limit:
                self.planning_trace.append(row)
        return row

    def block(self, truth, belief, action, start, duration, rng_truth, rng_pf,
              phase, *, scenario=False, collect=False):
        deaths = icu = social = 0.
        observations, states = [], []
        for day in range(duration):
            dt = start+timedelta(days=day)
            truth, belief, y, metrics, diagnostic, triggers = self.engine.step_with_filter(
                truth, belief, action, dt, rng_truth, rng_pf, phase=phase,
                oracle=self.config.oracle, scenario=scenario)
            d = float(metrics["deaths_total"][0]); u = float(metrics["icu"][0])
            price = self.engine.contact_loss(action)*self.engine.cost_scale if dt <= CONTROL_END else 0.
            deaths += d; icu += u; social += price
            state = self.record_state(belief, dt, phase, triggers, diagnostic)
            if collect:
                row = {"date": str(dt), "deaths": d, "icu_occupancy": u,
                       "social_cost": price, **{f"a_{a}": float(v) for a, v in zip(AGES, action)}}
                row.update({column: float(value) for column, value in zip(self.engine.obs_columns, y)})
                observations.append(row); states.append(state)
        return truth, belief, deaths, icu, social, observations, states

    def tail(self, truth, belief, rng_truth, rng_pf, phase, *, scenario=False, collect=False):
        dd = uu = 0.; rows = []; states = []
        for timestamp in pd.date_range("2021-01-16", "2021-03-12"):
            dt = timestamp.date(); action = self.engine.government_action(dt)
            truth, belief, d, u, _, r, s = self.block(truth, belief, action, dt, 1,
                rng_truth, rng_pf, phase, scenario=scenario, collect=collect)
            dd += d; uu += u; rows.extend(r); states.extend(s)
        return truth, belief, dd, uu, rows, states

    def government_reference(self):
        rng_t, rng_p = [np.random.default_rng(s) for s in np.random.SeedSequence([self.config.seed, 100]).spawn(2)]
        belief = self.initial.copy(); truth = self.engine.sample_truth(belief, rng_t)
        rows = []; states = []; d = u = 0.
        for timestamp in pd.date_range(START, END):
            dt = timestamp.date(); action = self.engine.government_action(dt)
            truth, belief, dd, uu, _, r, s = self.block(truth, belief, action, dt, 1, rng_t, rng_p,
                "deployment", scenario=True, collect=True)
            rows.extend(r); states.extend(s); d += dd; u += uu
        pending = self.pending_deaths(truth, END)
        self.references = max(d+pending, 1.), max(u, 1.)
        pd.DataFrame(rows).to_csv(self.output/"government_daily.csv", index=False)
        pd.DataFrame(states).to_csv(self.output/"government_belief_states.csv", index=False)
        return {"deaths_in_period": d, "icu_patient_days": u, "pending_expected_deaths_after_period": pending}

    def mc_value(self, current_belief, candidate, decision, history, spent, lam, seed):
        # A fresh hypothetical truth is sampled from the CURRENT PF belief.
        # Planning never receives the deployment world's hidden/future path.
        r_b, r_t, r_p = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)]
        belief = self.sample_belief(current_belief, r_b)
        truth = self.engine.sample_truth(belief, r_t)
        history = list(history); d = u = c = 0.
        start_day = sum(DECISION_LENGTHS[:decision])
        for j in range(decision, 7):
            feature = self.engine.features(belief, history, 96-start_day, spent+c)
            aid = candidate if j == decision else int(np.argmax(self.agent.values(feature)))
            action = ACTIONS[aid]
            truth, belief, dd, uu, cc, _, _ = self.block(truth, belief, action,
                START+timedelta(days=start_day), DECISION_LENGTHS[j], r_t, r_p, "planning")
            d += dd; u += uu; c += cc; history.append(action); start_day += DECISION_LENGTHS[j]
        truth, belief, dd, uu, _, _ = self.tail(truth, belief, r_t, r_p, "planning")
        d += dd+self.pending_deaths(truth, END); u += uu
        return self.health_reward(d, u, c, lam)

    def select_mc(self, belief, decision, history, spent, lam, iteration):
        feature = self.engine.features(belief, history, 96-sum(DECISION_LENGTHS[:decision]), spent)
        values = self.agent.values(feature)
        candidates = np.argsort(values, kind="stable")[-self.config.mc_candidates:]
        scores = []
        for candidate in candidates:
            scores.append(np.mean([self.mc_value(belief, int(candidate), decision, history,
                spent, lam, [self.config.seed, iteration, decision, rep, 771])
                for rep in range(self.config.mc_rollouts)]))
        return int(candidates[int(np.argmax(scores))]), float(max(scores))

    def training_episode(self, iteration, episode, lam):
        rng, r_t, r_p = [np.random.default_rng(s) for s in np.random.SeedSequence(
            [self.config.seed, iteration, episode, 331]).spawn(3)]
        belief = self.sample_belief(self.initial, rng)
        truth = self.engine.sample_truth(belief, r_t)
        history = []; spent = 0.; elapsed = 0
        for j, duration in enumerate(DECISION_LENGTHS):
            features = self.engine.features(belief, history, 96-elapsed, spent)
            epsilon = max(.10, .8*(1-episode/max(self.config.train_episodes, 1)))
            aid = self.agent.choose(features, rng, epsilon)
            action = ACTIONS[aid]
            truth, belief, d, u, cost, _, _ = self.block(truth, belief, action,
                START+timedelta(days=elapsed), duration, r_t, r_p, "planning")
            spent += cost; elapsed += duration; history.append(action)
            terminal = j == 6
            if terminal:
                truth, belief, dd, uu, _, _ = self.tail(truth, belief, r_t, r_p, "planning")
                d += dd+self.pending_deaths(truth, END); u += uu
            successor = self.engine.features(belief, history, 96-elapsed, spent)
            self.agent.remember(features, aid, self.health_reward(d, u, 0., 0.), successor,
                                terminal, self.health_reward(0., 0., cost, 1.))
            self.agent.learn(rng, lam)

    def deployment(self, iteration, lam):
        r_t, r_p = [np.random.default_rng(s) for s in np.random.SeedSequence([self.config.seed, 991]).spawn(2)]
        belief = self.initial.copy(); truth = self.engine.sample_truth(belief, r_t)
        spent = 0.; elapsed = 0; history = []; rows = []; states = []; decisions = []
        for j, duration in enumerate(DECISION_LENGTHS):
            feature = self.engine.features(belief, history, 96-elapsed, spent)
            aid, value = self.select_mc(belief, j, history, spent, lam, iteration)
            action = ACTIONS[aid]
            decision = {"iteration": iteration, "decision": j+1,
                        "date": str(START+timedelta(days=elapsed)), "lambda": lam,
                        "action_id": aid, "mc_value": value,
                        **{name: float(v) for name, v in zip(self.engine.feature_names(), feature)}}
            decision.update({f"a_{a}": float(v) for a, v in zip(AGES, action)})
            decisions.append(decision)
            truth, belief, _, _, cost, rr, ss = self.block(truth, belief, action,
                START+timedelta(days=elapsed), duration, r_t, r_p, "deployment", scenario=True, collect=True)
            elapsed += duration; spent += cost; history.append(action); rows.extend(rr); states.extend(ss)
        truth, belief, _, _, rr, ss = self.tail(truth, belief, r_t, r_p, "deployment", scenario=True, collect=True)
        rows.extend(rr); states.extend(ss)
        folder = self.output/f"iter_{iteration:02d}"
        folder.mkdir(exist_ok=True)
        pd.DataFrame(rows).to_csv(folder/"rl_daily.csv", index=False)
        pd.DataFrame(states).to_csv(folder/"belief_states.csv", index=False)
        pd.DataFrame(decisions).to_csv(folder/"decisions_and_policy_state.csv", index=False)
        self.agent.save(folder/"ddqn.pt", self.engine.feature_names())
        return spent, rows

    def run(self):
        (self.output/"feature_schema.json").write_text(json.dumps(self.engine.feature_names(), indent=2))
        references = self.government_reference()
        lam = self.config.lambda0 if self.config.lambda0 is not None else (.5 if self.engine.country == "england" else 1.6)
        trace = []; last_rows = None
        for iteration in range(1, self.config.outer_iterations+1):
            for episode in range(self.config.train_episodes):
                self.training_episode(iteration, episode, lam)
            spent, last_rows = self.deployment(iteration, lam)
            eta = self.config.eta0/np.sqrt(iteration)
            following = float(np.clip(lam+eta*(spent/self.engine.budget-1), 0, 4))
            trace.append({"iteration": iteration, "lambda_used": lam, "lambda_next": following,
                          "social_spend": spent, "budget": self.engine.budget, "eta": eta})
            pd.DataFrame(trace).to_csv(self.output/"dual_trace.csv", index=False)
            print(f"{self.engine.country}: completed iteration {iteration}, PF planning updates={self.engine.counters['planning']['pf_updates']}", flush=True)
            lam = following
        pd.DataFrame(self.planning_trace).to_csv(self.output/"planning_belief_states.csv", index=False)
        summary = {"country": self.engine.country, "config": asdict(self.config),
                   "social_budget": self.engine.budget,
                   "government_reference": references, "pf_counters": self.engine.counters,
                   "planning_trace_rows_seen": self._trace_seen, "planning_trace_rows_saved": len(self.planning_trace),
                   "network_updates": self.agent.updates,
                   "state_feature_count": len(self.engine.feature_names()),
                   "feature_statistics": ["posterior_median", "posterior_standard_deviation"],
                   "uses_current_simulated_truth_fallback": self.config.oracle,
                   "uses_future_deployment_truth_in_planning": False,
                   "uses_real_future_observations_in_counterfactual_filters": False,
                   "new_model_parameter_fit": False,
                   "method": "PF during training, hypothetical Monte Carlo planning and deployment; ensemble Double DQN",
                   "status": "smoke validation, not a scientific production result" if self.config.profile == "smoke" else "simulation research run"}
        if self.engine.counters["planning"]["pf_updates"] == 0 or self.agent.updates == 0:
            raise RuntimeError("Smoke did not exercise PF planning and network training")
        (self.output/"run_summary.json").write_text(json.dumps(summary, indent=2))
        return summary
