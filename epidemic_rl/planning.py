"""Original per-decision fitted-Q protocol, with PF median/SD rollout states.

Sources: supplied England V18 and Belgium V11/V12. Preserve episode counts,
per-decision retraining, fitted-Q, paired MC, independent evaluation and dual
updates. Replace full-state future features with particle-filter beliefs.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import timedelta
from itertools import product
import hashlib
import json
import os
from pathlib import Path
import pickle
import time

import numpy as np
import pandas as pd
import torch

from .engine import (Engine, START, CONTROL_END, CONTROL_DAYS, END, TAIL_START,
                     AGES, DECISION_LENGTHS, study_calendar)
from .fitted_q import Dataset, FittedPolicy, QNetwork, fit_ddqn
from .parallel import (atomic_pickle, digest_arrays, guard_checkpoint, ordered_map,
                       single_thread_worker, start_method, worker_pool)

ACTIONS = np.asarray(list(product((1., .7, .4, .2), repeat=3)))
ORIGINAL_EPISODES = (50000, 47000, 44000, 41000, 38000, 34000, 30000)
PROTOCOL = "original_per_decision_fqi_pf_median_sd_14day_v3"


@dataclass
class RunConfig:
    profile: str = "smoke"
    seed: int = 20261118
    deployment_particles: int = 32
    planning_particles: int = 16
    episodes_by_decision: tuple = (2,)*7
    outer_iterations: int = 1
    ensemble_size: int = 1
    fqi_iterations: int = 2
    epochs_per_iteration: int = 1
    mc_candidates: int = 2
    mc_rollouts: int = 1
    eval_episodes: int = 2
    eval_mc_rollouts: int = 1
    workers: int = 1
    fit_workers: int = 1
    mc_workers: int = 1
    eval_workers: int = 1
    batch_size: int = 512
    learning_rate: float = .0003
    weight_decay: float = 1e-5
    target_sync_every: int = 2
    grad_clip_norm: float = 10.
    alpha: float = .5
    lambda0: float | None = None
    eta0: float = .4
    oracle: bool = True
    planning_trace_limit: int = 3000
    episode_chunk_size: int = 512
    government_replicates: int = 1
    start_method: str = "auto"

    @classmethod
    def smoke(cls, country="england", parallel=False):
        config = cls()
        if country == "belgium":
            config.episodes_by_decision = (8,)*7
            config.batch_size, config.weight_decay, config.target_sync_every = 2048, 1e-4, 4
            config.episode_chunk_size, config.government_replicates = 100, 2
        if parallel:
            config.profile = "parallel-smoke"
            config.episodes_by_decision = (2,)*7
            config.ensemble_size = 3
            config.workers = config.fit_workers = config.mc_workers = config.eval_workers = 2
        return config

    @classmethod
    def research(cls, country="england"):
        config = cls.smoke(country)
        config.profile = "research"
        config.deployment_particles, config.planning_particles = 5000, 1024
        config.episodes_by_decision = ORIGINAL_EPISODES
        config.outer_iterations, config.ensemble_size = 20, 3
        config.fqi_iterations, config.epochs_per_iteration = 36, 3
        config.mc_candidates, config.mc_rollouts = 64, 192
        config.eval_episodes, config.eval_mc_rollouts = 16, 32
        config.workers = config.mc_workers = config.eval_workers = 8
        config.fit_workers, config.planning_trace_limit = 3, 10000
        config.government_replicates = 12 if country == "belgium" else 1
        return config

    def fit_config(self):
        names = ("ensemble_size", "fqi_iterations", "epochs_per_iteration", "batch_size",
                 "learning_rate", "weight_decay", "target_sync_every", "grad_clip_norm")
        return {name: getattr(self, name) for name in names}

    def validate(self):
        positive = ("deployment_particles", "planning_particles", "outer_iterations", "ensemble_size",
                    "fqi_iterations", "epochs_per_iteration", "mc_candidates", "mc_rollouts",
                    "eval_episodes", "eval_mc_rollouts", "workers", "fit_workers", "mc_workers",
                    "eval_workers", "batch_size", "target_sync_every", "episode_chunk_size",
                    "government_replicates")
        if any(getattr(self, name) < 1 for name in positive):
            raise ValueError("Particle, episode, iteration and worker counts must be positive")
        if len(self.episodes_by_decision) != 7 or any(n < 1 for n in self.episodes_by_decision):
            raise ValueError("Exactly seven positive per-decision episode counts are required")
        if self.mc_candidates > 64 or not 0 <= self.alpha <= 1:
            raise ValueError("Invalid candidates or alpha")
        start_method(self.start_method)


def scientific_config(config):
    values = asdict(config)
    for key in ("workers", "fit_workers", "mc_workers", "eval_workers", "start_method"):
        values.pop(key)
    return values


def pending_deaths(engine, truth):
    model = engine.model(END)
    community, ward, icu = engine.mortality
    hd = (1-model.p_icu)*ward + model.p_icu*icu
    ifr = (1-model.p_hospital)*community + model.p_hospital*hd
    x = truth.x[0]
    return float((x[:, 1:5].sum(1)*ifr + x[:, 5]*hd + x[:, 6]*icu).sum())


def health_reward(deaths, icu, social, lam, references, budget, alpha):
    return -CONTROL_DAYS*(alpha*deaths/references[0] + (1-alpha)*icu/references[1] + lam*social/budget)


def dual_update(lam, expected_spend, budget, eta0, iteration):
    eta = eta0/np.sqrt(iteration)
    return float(np.clip(lam + eta*(expected_spend/budget-1), 0., 4.)), float(eta)


def reset_counters(engine):
    for values in engine.counters.values():
        for key in values:
            values[key] = 0


class Simulation:
    def __init__(self, engine, config, trace_limit=0, role="planning", metadata=None):
        self.engine, self.config = engine, config
        self.trace, self.trace_seen, self.trace_limit = [], 0, trace_limit
        self.role, self.metadata = role, (metadata or {})

    def block(self, truth, belief, action, start, duration, rng_t, rng_p,
              phase="planning", scenario=False, collect=False):
        deaths = icu = social = 0.
        rows, states = [], []
        for day in range(duration):
            dt = start+timedelta(days=day)
            truth, belief, y, metrics, diagnostic, triggers = self.engine.step_with_filter(
                truth, belief, action, dt, rng_t, rng_p, phase=phase,
                oracle=self.config.oracle, scenario=scenario)
            d, u = float(metrics["deaths_total"][0]), float(metrics["icu"][0])
            cost = self.engine.contact_loss(action)*self.engine.cost_scale if dt <= CONTROL_END else 0.
            deaths += d; icu += u; social += cost
            if phase == "planning":
                self.trace_seen += 1
            keep = phase == "planning" and (self.trace_limit < 0 or len(self.trace) < self.trace_limit)
            if collect or keep:
                state = {"date": str(dt), "phase": phase, "role": self.role,
                         "oracle_reset": bool(triggers), "oracle_triggers": ";".join(triggers),
                         **diagnostic, **self.metadata, **self.engine.state_row(belief)}
                if keep:
                    self.trace.append(state)
                if collect:
                    states.append(state)
            if collect:
                rows.append({"date": str(dt), "deaths": d, "icu_occupancy": u, "social_cost": cost,
                             **{f"a_{a}": float(v) for a, v in zip(AGES, action)},
                             **{name: float(value) for name, value in zip(self.engine.obs_columns, y)}})
        return truth, belief, deaths, icu, social, rows, states

    def tail(self, truth, belief, rng_t, rng_p, phase="planning", scenario=False, collect=False):
        d = u = 0.; rows, states = [], []
        for timestamp in pd.date_range(TAIL_START, END):
            dt = timestamp.date()
            truth, belief, dd, uu, _, r, s = self.block(truth, belief,
                self.engine.government_action(dt), dt, 1, rng_t, rng_p, phase, scenario, collect)
            d += dd; u += uu; rows.extend(r); states.extend(s)
        return truth, belief, d, u, rows, states


def sample_root(engine, belief, config, rng_b, rng_t):
    idx = rng_b.choice(belief.n, size=config.planning_particles, replace=True, p=belief.weights)
    return engine.sample_truth(belief, rng_t), belief.subset(idx)


_TRAIN = None


def _train_init(context):
    global _TRAIN
    single_thread_worker()
    _TRAIN = context


def _training_episode(task):
    episode, trace_limit = task
    c = _TRAIN; engine, cfg = c["engine"], c["config"]
    reset_counters(engine)
    rng_b, rng_a, rng_t, rng_p = [np.random.default_rng(s) for s in
        np.random.SeedSequence([c["seed"], episode, c["decision"], 331]).spawn(4)]
    truth, belief = sample_root(engine, c["belief"], cfg, rng_b, rng_t)
    sim = Simulation(engine, cfg, trace_limit, "training", {"episode": episode,
                     "origin_decision": c["decision"]+1})
    history, spent, rows = list(c["history"]), c["spent"], []
    for decision in range(c["decision"], 7):
        elapsed = sum(DECISION_LENGTHS[:decision])
        state = engine.features(belief, history, CONTROL_DAYS-elapsed, spent)
        # Original exploration: balanced root actions in England, uniformly
        # random subsequent actions; Belgium samples uniformly at every stage.
        aid = episode % 64 if engine.country == "england" and decision == c["decision"] else int(rng_a.integers(64))
        action = ACTIONS[aid]
        truth, belief, d, u, cost, _, _ = sim.block(truth, belief, action,
            START+timedelta(days=elapsed), DECISION_LENGTHS[decision], rng_t, rng_p)
        spent += cost; history.append(action)
        done = decision == 6
        if done:
            truth, belief, dd, uu, _, _ = sim.tail(truth, belief, rng_t, rng_p)
            d += dd+pending_deaths(engine, truth); u += uu
            successor, mask = np.zeros(len(state), np.float32), np.zeros(64, bool)
        else:
            successor = engine.features(belief, history, CONTROL_DAYS-sum(DECISION_LENGTHS[:decision+1]), spent)
            mask = np.ones(64, bool)
        reward = health_reward(d, u, cost, c["lam"], c["references"], engine.budget, cfg.alpha)
        rows.append((state, aid, reward, successor, float(done), mask, episode, decision))
    return {"rows": rows, "counters": engine.counters, "trace": sim.trace,
            "trace_seen": sim.trace_seen, "pid": os.getpid()}


def _context_digest(context, extra=None):
    belief = context["belief"]
    metadata = {key: context[key] for key in ("decision", "spent", "history", "lam", "references", "seed") if key in context}
    metadata.update(config=scientific_config(context["config"]), protocol=PROTOCOL,
                    country=context["engine"].country, extra=extra, code=context["code_digest"])
    return digest_arrays(belief.fingerprint_arrays(), metadata)


def generate_dataset(context, episodes, workers=1, checkpoint=None, trace_limit=0):
    cfg = context["config"]; decision = context["decision"]
    directory = guard_checkpoint(checkpoint, _context_digest(context, {"episodes": episodes})) if checkpoint is not None else None
    rows, summaries, trace = [], [], []
    remaining_days = (END-(START+timedelta(days=sum(DECISION_LENGTHS[:decision])))).days+1
    # One persistent pool for all checkpoint chunks of this decision.
    with worker_pool(min(workers, episodes), _train_init, context, cfg.start_method) as pool:
        for lo in range(0, episodes, cfg.episode_chunk_size):
            hi = min(lo+cfg.episode_chunk_size, episodes)
            path = directory/f"chunk_{lo:07d}_{hi:07d}.pkl" if directory is not None else None
            if path is not None and path.exists():
                chunk = pickle.loads(path.read_bytes())
            else:
                tasks = [(i, -1 if trace_limit < 0 else max(0, min(remaining_days, trace_limit-i*remaining_days))) for i in range(lo, hi)]
                chunk = [pickle.loads(pickle.dumps(r)) for r in ordered_map(pool, _training_episode, tasks)]
                if path is not None:
                    atomic_pickle(path, chunk)
            if len(chunk) != hi-lo or sum(len(r["rows"]) for r in chunk) != (hi-lo)*(7-decision):
                raise RuntimeError("Incomplete per-decision training checkpoint")
            for result in chunk:
                rows.extend(result.pop("rows")); trace.extend(result.pop("trace")); summaries.append(result)
            print(f"TRAIN D{decision+1}: episodes {hi}/{episodes}, transitions {len(rows)}/{episodes*(7-decision)}", flush=True)
    columns = list(zip(*rows))
    dataset = Dataset(np.asarray(columns[0], np.float32), np.asarray(columns[1], np.int64),
                      np.asarray(columns[2], np.float32), np.asarray(columns[3], np.float32),
                      np.asarray(columns[4], np.float32), np.asarray(columns[5], bool),
                      np.asarray(columns[6], np.int32), np.asarray(columns[7], np.int16))
    return dataset, summaries, trace


_MC = None


def _mc_init(context):
    global _MC
    single_thread_worker()
    _MC = dict(context, policy=FittedPolicy.from_payload(context["policy"]))


def _mc_action(task):
    aid, trace_limit = task
    c = _MC; engine, cfg = c["engine"], c["config"]
    reset_counters(engine)
    values, spends, trace, seen = [], [], [], 0
    for rep in range(c["rollouts"]):
        # Candidate action is absent from the seed: paired random numbers.
        rng_b, rng_t, rng_p = [np.random.default_rng(s) for s in
            np.random.SeedSequence([c["seed"], rep, 771]).spawn(3)]
        truth, belief = sample_root(engine, c["belief"], cfg, rng_b, rng_t)
        sim = Simulation(engine, cfg, -1 if trace_limit < 0 else max(0, trace_limit-len(trace)),
                         "monte_carlo", {"candidate_action": aid, "rollout": rep,
                                          "origin_decision": c["decision"]+1})
        history, spent, d, u = list(c["history"]), c["spent"], 0., 0.
        for decision in range(c["decision"], 7):
            elapsed = sum(DECISION_LENGTHS[:decision])
            choice = aid if decision == c["decision"] else int(np.argmax(c["policy"].values(
                engine.features(belief, history, CONTROL_DAYS-elapsed, spent))))
            action = ACTIONS[choice]
            truth, belief, dd, uu, cost, _, _ = sim.block(truth, belief, action,
                START+timedelta(days=elapsed), DECISION_LENGTHS[decision], rng_t, rng_p)
            d += dd; u += uu; spent += cost; history.append(action)
        truth, belief, dd, uu, _, _ = sim.tail(truth, belief, rng_t, rng_p)
        d += dd+pending_deaths(engine, truth); u += uu
        values.append(health_reward(d, u, spent-c["spent"], c["lam"], c["references"], engine.budget, cfg.alpha))
        spends.append(spent); trace.extend(sim.trace); seen += sim.trace_seen
    count = c["rollouts"]
    row = {"action_id": aid, "mean_mc_return": float(np.mean(values)),
           "sd_mc_return": float(np.std(values, ddof=1)) if count > 1 else 0.,
           "mean_projected_total_budget": float(np.mean(spends)), "n_rollouts": count}
    return {"row": row, "counters": engine.counters, "trace": trace, "trace_seen": seen, "pid": os.getpid()}


def select_mc(context, workers, trace_limit=0):
    policy = FittedPolicy.from_payload(context["policy"]); engine = context["engine"]
    state = engine.features(context["belief"], context["history"],
                            CONTROL_DAYS-sum(DECISION_LENGTHS[:context["decision"]]), context["spent"])
    q, qsd = policy.values(state, return_std=True)
    candidates = np.argsort(-q, kind="stable")[:context["config"].mc_candidates]
    days = (END-(START+timedelta(days=sum(DECISION_LENGTHS[:context["decision"]])))).days+1
    tasks = [(int(aid), -1 if trace_limit < 0 else max(0, min(days*context["rollouts"], trace_limit-i*days*context["rollouts"]))) for i, aid in enumerate(candidates)]
    with worker_pool(min(workers, len(tasks)), _mc_init, context, context["config"].start_method) as pool:
        results = [pickle.loads(pickle.dumps(r)) for r in ordered_map(pool, _mc_action, tasks)]
    for result in results:
        aid = result["row"]["action_id"]
        result["row"].update(ddqn_Q=float(q[aid]), ensemble_sd=float(qsd[aid]))
    rank_mc_results(results, engine.country)
    return results[0]["row"]["action_id"], results


def rank_mc_results(results, country):
    # England's source sorts (-return, action_id); Belgium's max(return,
    # action_id, sd) picks the larger action ID when returns are exactly tied.
    sign = 1 if country == "england" else -1
    results.sort(key=lambda result: (-result["row"]["mean_mc_return"], sign*result["row"]["action_id"]))


_EVAL = None


def _eval_init(context):
    global _EVAL
    single_thread_worker(); _EVAL = context


def _evaluate_policy(rep):
    c = _EVAL; engine, cfg = c["engine"], c["config"]
    reset_counters(engine)
    rng_b, rng_t, rng_p = [np.random.default_rng(s) for s in
        np.random.SeedSequence([cfg.seed, c["iteration"], rep, 12345]).spawn(3)]
    belief = c["belief"].subset(rng_b.choice(c["belief"].n, size=cfg.deployment_particles,
                                            replace=True, p=c["belief"].weights))
    truth = engine.sample_truth(c["belief"], rng_t)
    sim = Simulation(engine, cfg, role="independent_evaluation")
    history, spent, d, u, choices, statistics = [], 0., 0., 0., [], []
    for decision, duration in enumerate(DECISION_LENGTHS):
        elapsed = sum(DECISION_LENGTHS[:decision])
        belief, _ = engine.oracle_gate(belief, truth, enabled=cfg.oracle, phase="evaluation")
        statistics.append({"counters": pickle.loads(pickle.dumps(engine.counters)), "trace_seen": 0, "pid": os.getpid()})
        context = dict(c, belief=belief, decision=decision, spent=spent, history=history,
                       policy=c["policies"][decision], rollouts=cfg.eval_mc_rollouts,
                       seed=cfg.seed+c["iteration"]*891000+rep*1009+decision*17)
        # This worker belongs to the OUTER evaluation pool: MC is serial here.
        aid, results = select_mc(context, workers=1, trace_limit=0)
        statistics.extend({k: result[k] for k in ("counters", "trace_seen", "pid")} for result in results)
        reset_counters(engine)
        action = ACTIONS[aid]
        truth, belief, dd, uu, cost, _, _ = sim.block(truth, belief, action,
            START+timedelta(days=elapsed), duration, rng_t, rng_p, phase="evaluation", scenario=True)
        d += dd; u += uu; spent += cost; history.append(action)
        choices.append({"decision": decision+1, "action_id": aid, "spent": spent})
        statistics.append({"counters": pickle.loads(pickle.dumps(engine.counters)), "trace_seen": 0, "pid": os.getpid()})
        reset_counters(engine)
    truth, belief, dd, uu, _, _ = sim.tail(truth, belief, rng_t, rng_p, phase="evaluation", scenario=True)
    d += dd+pending_deaths(engine, truth); u += uu
    statistics.append({"counters": engine.counters, "trace_seen": 0, "pid": os.getpid()})
    result = {"evaluation_episode": rep, "social_cost": spent, "deaths_plus_pending": d,
              "icu_patient_days": u, "actions": choices, "mc_workers_inside_evaluation": 1,
              "pid": os.getpid(), "statistics": statistics}
    atomic_pickle(c["checkpoint"]/f"rep_{rep:04d}.pkl", result)
    return result


class Experiment:
    def __init__(self, country, config, output):
        config.validate(); single_thread_worker()
        self.config, self.output = config, Path(output); self.output.mkdir(parents=True, exist_ok=True)
        self.engine = Engine(country)
        self.initial = self.engine.initial_belief(config.deployment_particles, config.seed)
        self.references = None
        self.stats = {"network_updates": 0, "fits": 0, "planning_trace_seen": 0,
                      "worker_pids": {name: [] for name in ("training", "fit", "mc", "evaluation")}}
        self.counters = pickle.loads(pickle.dumps(self.engine.counters)); self.planning_trace = []
        self.code_digest = hashlib.sha256(b"".join((Path(__file__).parent/name).read_bytes() for name in
            ("planning.py", "parallel.py", "fitted_q.py", "engine.py", f"parameters_{country}.py"))).hexdigest()
        arrays = self.initial.fingerprint_arrays()
        self.digest = digest_arrays(arrays, {"config": scientific_config(config), "country": country,
            "code": self.code_digest, "inputs": hashlib.sha256((self.engine.data_dir/'pf_observations.csv').read_bytes()).hexdigest()})
        guard_checkpoint(self.output, self.digest)

    def remaining_trace(self):
        limit = self.config.planning_trace_limit
        return -1 if limit < 0 else max(0, limit-len(self.planning_trace))

    def absorb(self, result, phase=None):
        for name, values in result.get("counters", {}).items():
            for key, value in values.items():
                self.counters[name][key] += value
        self.stats["planning_trace_seen"] += result.get("trace_seen", 0)
        if phase is not None:
            self.stats["worker_pids"][phase] = sorted(set(self.stats["worker_pids"][phase]+[result["pid"]]))
        limit = self.remaining_trace(); trace = result.get("trace", [])
        self.planning_trace.extend(trace if limit < 0 else trace[:limit])

    def context(self, belief, decision, history, spent, lam, seed):
        return dict(engine=self.engine, config=self.config, belief=belief, decision=decision,
                    history=history, spent=spent, lam=lam, references=self.references,
                    seed=int(seed), code_digest=self.code_digest)

    def government_reference(self):
        checkpoint = self.output/'government_reference.pkl'
        if checkpoint.exists():
            result = pickle.loads(checkpoint.read_bytes())
            self.references, self.counters = tuple(result["health_denominators"]), result["counters"]
            return result["summary"]
        observations, deaths, icu, pending = [], [], [], []
        for rep in range(self.config.government_replicates):
            reset_counters(self.engine)
            rng_t, rng_p = [np.random.default_rng(s) for s in np.random.SeedSequence([self.config.seed, 100, rep]).spawn(2)]
            belief = self.initial.copy(); truth = self.engine.sample_truth(belief, rng_t)
            sim = Simulation(self.engine, self.config, role="government")
            rows, states, d, u = [], [], 0., 0.
            for timestamp in pd.date_range(START, END):
                dt = timestamp.date()
                truth, belief, dd, uu, _, rr, ss = sim.block(truth, belief,
                    self.engine.government_action(dt), dt, 1, rng_t, rng_p,
                    phase="deployment", scenario=True, collect=(rep == 0))
                rows.extend(rr); states.extend(ss); d += dd; u += uu
            deaths.append(d); icu.append(u); pending.append(pending_deaths(self.engine, truth))
            self.absorb({"counters": self.engine.counters, "pid": os.getpid()})
            observations.append({"replicate": rep, "deaths_in_period": d,
                                  "icu_patient_days": u, "pending_expected_deaths_after_period": pending[-1]})
            if rep == 0:
                pd.DataFrame(rows).to_csv(self.output/'government_daily.csv', index=False)
                pd.DataFrame(states).to_csv(self.output/'government_belief_states.csv', index=False)
        self.references = max(float(np.mean(np.array(deaths)+pending)), 1.), max(float(np.mean(icu)), 1.)
        summary = {"deaths_in_period": float(np.mean(deaths)), "icu_patient_days": float(np.mean(icu)),
                   "pending_expected_deaths_after_period": float(np.mean(pending)),
                   "replicates": self.config.government_replicates}
        pd.DataFrame(observations).to_csv(self.output/'government_reference_replicates.csv', index=False)
        atomic_pickle(checkpoint, {"summary": summary, "health_denominators": self.references,
                                  "counters": pickle.loads(pickle.dumps(self.counters))})
        return summary

    def deployment(self, iteration, lam):
        folder = self.output/f"iter_{iteration:02d}"; folder.mkdir(exist_ok=True)
        checkpoint = folder/'deployment_checkpoint.pkl'
        if checkpoint.exists():
            saved = pickle.loads(checkpoint.read_bytes())
            belief, truth, rng_t, rng_p = (saved[key] for key in ("belief", "truth", "rng_t", "rng_p"))
            spent, history, rows, states, decisions, policies, timings = (saved[key] for key in
                ("spent", "history", "rows", "states", "decisions", "policies", "timings"))
            self.stats, self.counters, self.planning_trace = (saved[key] for key in ("stats", "counters", "trace"))
        else:
            rng_t, rng_p = [np.random.default_rng(s) for s in np.random.SeedSequence([self.config.seed, 991]).spawn(2)]
            belief = self.initial.copy(); truth = self.engine.sample_truth(belief, rng_t)
            spent, history, rows, states, decisions, policies, timings = 0., [], [], [], [], [], []
        for decision in range(len(decisions), 7):
            elapsed = sum(DECISION_LENGTHS[:decision]); duration = DECISION_LENGTHS[decision]
            reset_counters(self.engine)
            belief, _ = self.engine.oracle_gate(belief, truth, enabled=self.config.oracle, phase="deployment")
            self.absorb({"counters": self.engine.counters, "pid": os.getpid()})
            context = self.context(belief, decision, history, spent, lam,
                                   self.config.seed+iteration*70000003+decision*8009)
            directory = folder/f"decision_{decision+1:02d}"; directory.mkdir(exist_ok=True)
            episodes = self.config.episodes_by_decision[decision]
            print(f"ITER {iteration} D{decision+1}/7: episodes={episodes}, lambda={lam:.6f}, "
                  f"workers={self.config.workers}/{self.config.fit_workers}/{self.config.mc_workers}", flush=True)
            tic = time.monotonic()
            dataset, summaries, trace = generate_dataset(context, episodes, self.config.workers,
                directory/'training_chunks', self.remaining_trace())
            training_seconds = time.monotonic()-tic
            for result in summaries:
                self.absorb(result, "training")
            limit = self.remaining_trace(); self.planning_trace.extend(trace if limit < 0 else trace[:limit])
            if len(dataset) != episodes*(7-decision) or dataset.states.shape[1] != len(self.engine.feature_names()):
                raise RuntimeError("Incomplete training dataset or incorrect PF feature schema")
            tic = time.monotonic()
            policy, logs, fit_audit = fit_ddqn(dataset, self.config.fit_config(), context["seed"]+71203,
                self.engine.country, self.config.fit_workers, directory/'fit_members', self.config.start_method)
            fit_seconds = time.monotonic()-tic
            self.stats["network_updates"] += fit_audit["updates"]; self.stats["fits"] += 1
            self.stats["worker_pids"]["fit"] = sorted(set(self.stats["worker_pids"]["fit"]+fit_audit["worker_pids"]))
            policies.append(policy.payload()); atomic_pickle(directory/'policy.pkl', policy.payload())
            torch.save({"state_dicts": [net.state_dict() for net in policy.nets], "scaler": policy.scaler,
                        "feature_names": self.engine.feature_names(), "fit_config": self.config.fit_config()}, directory/'ddqn.pt')
            pd.DataFrame(logs).to_csv(directory/'ddqn_log.csv', index=False)
            context.update(policy=policy.payload(), rollouts=self.config.mc_rollouts)
            tic = time.monotonic(); aid, results = select_mc(context, self.config.mc_workers, self.remaining_trace())
            mc_seconds = time.monotonic()-tic
            for result in results:
                self.absorb(result, "mc")
            pd.DataFrame([result["row"] for result in results]).to_csv(directory/'mc_reranking.csv', index=False)
            feature = self.engine.features(belief, history, CONTROL_DAYS-elapsed, spent); action = ACTIONS[aid]
            decisions.append({"iteration": iteration, "decision": decision+1, "date": str(START+timedelta(days=elapsed)),
                "end_date": str(START+timedelta(days=elapsed+duration-1)), "duration_days": duration,
                "lambda": lam, "action_id": aid, "mc_value": results[0]["row"]["mean_mc_return"],
                "oracle_point_corrected": belief.oracle_state is not None,
                **{name: float(value) for name, value in zip(self.engine.feature_names(), feature)},
                **{f"a_{age}": float(value) for age, value in zip(AGES, action)}})
            reset_counters(self.engine); sim = Simulation(self.engine, self.config, role="main_deployment")
            truth, belief, _, _, cost, rr, ss = sim.block(truth, belief, action,
                START+timedelta(days=elapsed), duration, rng_t, rng_p,
                phase="deployment", scenario=True, collect=True)
            self.absorb({"counters": self.engine.counters, "pid": os.getpid()})
            spent += cost; history.append(action); rows.extend(rr); states.extend(ss)
            timing = {"decision": decision+1, "training_episodes": episodes,
                "training_transitions": len(dataset), "training_workers": self.config.workers,
                "fit_workers": self.config.fit_workers, "mc_workers": self.config.mc_workers,
                "training_pids": sorted({r["pid"] for r in summaries}), "fit_pids": fit_audit["worker_pids"],
                "mc_pids": sorted({r["pid"] for r in results}), "training_seconds": training_seconds,
                "fit_seconds": fit_seconds, "mc_seconds": mc_seconds, "fit_config": self.config.fit_config()}
            timings.append(timing); (directory/'timings.json').write_text(json.dumps(timing, indent=2))
            pd.DataFrame(rows).to_csv(folder/'rl_daily.csv', index=False)
            pd.DataFrame(states).to_csv(folder/'belief_states.csv', index=False)
            pd.DataFrame(decisions).to_csv(folder/'decisions_and_policy_state.csv', index=False)
            atomic_pickle(checkpoint, {"belief": belief, "truth": truth, "rng_t": rng_t, "rng_p": rng_p,
                "spent": spent, "history": history, "rows": rows, "states": states, "decisions": decisions,
                "policies": policies, "timings": timings, "stats": self.stats, "counters": self.counters, "trace": self.planning_trace})
        complete = folder/'deployment_complete.pkl'
        if not complete.exists():
            reset_counters(self.engine); sim = Simulation(self.engine, self.config, role="main_deployment")
            truth, belief, _, _, rr, ss = sim.tail(truth, belief, rng_t, rng_p,
                phase="deployment", scenario=True, collect=True)
            rows.extend(rr); states.extend(ss)
            self.absorb({"counters": self.engine.counters, "pid": os.getpid()})
            pd.DataFrame(rows).to_csv(folder/'rl_daily.csv', index=False)
            pd.DataFrame(states).to_csv(folder/'belief_states.csv', index=False)
            atomic_pickle(complete, {"spent": spent, "policies": policies,
                "stats": self.stats, "counters": self.counters, "trace": self.planning_trace})
        else:
            saved = pickle.loads(complete.read_bytes())
            self.stats, self.counters, self.planning_trace = (saved[key] for key in ("stats", "counters", "trace"))
        return spent, policies

    def evaluate(self, iteration, lam, policies):
        directory = self.output/f"iter_{iteration:02d}"/'evaluation_checkpoint'
        context = self.context(self.initial, 0, [], 0., lam, self.config.seed+iteration*891000)
        context.update(policies=policies, iteration=iteration, checkpoint=directory)
        guard_checkpoint(directory, _context_digest(context, {"policies": hashlib.sha256(pickle.dumps(policies)).hexdigest()}))
        results = {}
        for rep in range(self.config.eval_episodes):
            path = directory/f"rep_{rep:04d}.pkl"
            if path.exists():
                results[rep] = pickle.loads(path.read_bytes())
        pending = [rep for rep in range(self.config.eval_episodes) if rep not in results]
        if pending:
            with worker_pool(min(self.config.eval_workers, len(pending)), _eval_init, context, self.config.start_method) as pool:
                for result in ordered_map(pool, _evaluate_policy, pending):
                    results[result["evaluation_episode"]] = pickle.loads(pickle.dumps(result))
        rows = [results[rep] for rep in range(self.config.eval_episodes)]
        for result in rows:
            self.stats["worker_pids"]["evaluation"] = sorted(set(self.stats["worker_pids"]["evaluation"]+[result["pid"]]))
            for statistics in result["statistics"]:
                self.absorb(statistics, "evaluation")
        (self.output/f"iter_{iteration:02d}"/'independent_evaluations.json').write_text(json.dumps(rows, indent=2))
        return rows

    def run(self):
        (self.output/'feature_schema.json').write_text(json.dumps(self.engine.feature_names(), indent=2))
        references = self.government_reference()
        lam = self.config.lambda0 if self.config.lambda0 is not None else (.5 if self.engine.country == "england" else .8)
        trace = []
        for iteration in range(1, self.config.outer_iterations+1):
            spent, policies = self.deployment(iteration, lam)
            evaluations = self.evaluate(iteration, lam, policies)
            costs = np.array([row["social_cost"] for row in evaluations]); expected = float(costs.mean())
            following, eta = dual_update(lam, expected, self.engine.budget, self.config.eta0, iteration)
            trace.append({"iteration": iteration, "lambda_used": lam, "lambda_next": following,
                          "social_spend": spent, "expected_policy_spend": expected,
                          "evaluation_spend_standard_error": float(costs.std(ddof=1)/np.sqrt(len(costs))) if len(costs)>1 else 0.,
                          "evaluation_trajectories": len(evaluations), "budget": self.engine.budget, "eta": eta})
            pd.DataFrame(trace).to_csv(self.output/'dual_trace.csv', index=False); lam = following
            print(f"COMPLETED ITERATION {iteration}: realized spend={spent:.6f}, "
                  f"independent expected spend={expected:.6f}, lambda next={lam:.6f}", flush=True)
        pd.DataFrame(self.planning_trace).to_csv(self.output/'planning_belief_states.csv', index=False)
        summary = {"country": self.engine.country, "config": asdict(self.config), "protocol": PROTOCOL,
            "calendar": study_calendar(),
            "social_budget": self.engine.budget, "government_reference": references,
            "health_denominators": self.references, "pf_counters": self.counters,
            "planning_trace_rows_seen": self.stats["planning_trace_seen"], "planning_trace_rows_saved": len(self.planning_trace),
            "network_updates": self.stats["network_updates"], "per_decision_fits": self.stats["fits"],
            "worker_pids": self.stats["worker_pids"], "state_feature_count": len(self.engine.feature_names()),
            "feature_statistics": ["posterior_median", "posterior_standard_deviation"],
            "uses_current_simulated_truth_fallback": self.config.oracle,
            "oracle_correction_scope": "policy point estimates only; PF particles, weights and standard deviations retained",
            "oracle_changes_posterior_uncertainty": False,
            "uses_future_deployment_truth_in_planning": False,
            "uses_real_future_observations_in_counterfactual_filters": False,
            "new_model_parameter_fit": False, "dual_update_source": "independent closed-loop policy evaluations",
            "nested_evaluation_mc_workers": 1,
            "method": "Original per-decision bootstrapped Double-DQN fitted-Q; PF training/MC and median/SD features",
            "status": "simulation research run" if self.config.profile == "research" else "smoke validation, not a scientific production result"}
        if self.counters["planning"]["pf_updates"] == 0 or self.stats["network_updates"] == 0:
            raise RuntimeError("Run did not exercise PF planning and network fitting")
        (self.output/'run_summary.json').write_text(json.dumps(summary, indent=2))
        return summary
