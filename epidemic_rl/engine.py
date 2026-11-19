"""The existing eight-compartment stochastic model and country observation laws.

Ordered, exclusive binomial exits follow the archived England/Belgium model.
Particle filtering is used for both deployment and hypothetical planning paths.
No parameter estimation is performed in this module.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import date
from importlib import import_module
from pathlib import Path
import copy
import math

import numpy as np
import pandas as pd
from scipy.special import expit, logit, logsumexp
from scipy.stats import norm

from legacy.england_fit.reproduce import Model, initial_cloud, nb_logpmf, shift_history
from .paths import ROOT

AGES = ("0_44", "45_64", "65_plus")
COMPARTMENTS = ("S", "E1", "E2", "I1", "I2", "H", "U", "R")
S, E1, E2, I1, I2, H, U, R = range(8)
START, CONTROL_END, END = date(2020, 10, 12), date(2021, 1, 15), date(2021, 3, 12)
DECISION_LENGTHS = (14, 14, 14, 14, 14, 14, 12)


def systematic_resample(weights, rng):
    """One uniform offset, equally spaced positions, normalized cumulative mass."""
    w = np.asarray(weights, float)
    if w.ndim != 1 or len(w) == 0 or np.any(w < 0) or not np.isfinite(w).all() or w.sum() <= 0:
        raise ValueError("Invalid resampling weights")
    w = w / w.sum()
    cdf = np.cumsum(w)
    cdf[-1] = 1.0
    positions = (rng.random() + np.arange(len(w))) / len(w)
    return np.searchsorted(cdf, positions, side="right")


def weighted_summary(values, weights):
    """Weighted posterior median and population SD; supports arbitrary columns."""
    values = np.asarray(values, float)
    weights = np.asarray(weights, float)
    weights = weights / weights.sum()
    mean = weights @ values
    std = np.sqrt(np.maximum(weights @ np.square(values - mean), 0.0))
    order = np.argsort(values, axis=0, kind="stable")
    ordered = np.take_along_axis(values, order, axis=0)
    mass = np.cumsum(weights[order], axis=0)
    idx = np.argmax(mass >= 0.5, axis=0)
    median = ordered[idx, np.arange(values.shape[1])]
    return median, std


@dataclass
class Cloud:
    x: np.ndarray
    ch: np.ndarray
    hh: np.ndarray
    logq: np.ndarray
    zrho: np.ndarray
    eH: np.ndarray
    u_prev: np.ndarray
    mv_prev: np.ndarray
    deaths: np.ndarray
    logw: np.ndarray
    # Current-time privileged point estimate, separate from PF uncertainty.
    # Its particle arrays hold one simulated truth; they never replace this PF.
    oracle_state: Cloud | None = None

    def particle_arrays(self):
        return [(f.name, getattr(self, f.name)) for f in fields(self) if f.name != "oracle_state"]

    def fingerprint_arrays(self):
        arrays = [value for _, value in self.particle_arrays()]
        arrays.append(np.array([self.oracle_state is not None], dtype=np.uint8))
        if self.oracle_state is not None:
            arrays.extend(value for _, value in self.oracle_state.particle_arrays())
        return arrays

    @property
    def n(self):
        return len(self.logw)

    @property
    def weights(self):
        z = self.logw - logsumexp(self.logw)
        w = np.exp(z)
        if not np.isfinite(w).all():
            raise RuntimeError("Nonfinite PF weights")
        return w / w.sum()

    def copy(self):
        result = Cloud(*(value.copy() for _, value in self.particle_arrays()))
        if self.oracle_state is not None:
            result.oracle_state = self.oracle_state.copy()
        return result

    def subset(self, indices):
        result = Cloud(*(value[indices].copy() for _, value in self.particle_arrays()))
        result.logw[:] = -math.log(result.n)
        if self.oracle_state is not None:
            result.oracle_state = self.oracle_state.copy()
        return result

    def resample(self, indices):
        for name, value in self.particle_arrays():
            setattr(self, name, value[indices].copy())
        self.logw[:] = -math.log(self.n)
        self.oracle_state = None


class Engine:
    def __getstate__(self):
        # Importable parameter modules are reconstructed in Windows spawn
        # workers; transmitting a module object itself is not pickleable.
        state = self.__dict__.copy()
        state.pop("params")
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.params = import_module(f"epidemic_rl.parameters_{self.country}")

    def __init__(self, country):
        if country not in ("england", "belgium"):
            raise ValueError("Choose england or belgium")
        self.country = country
        self.params = import_module(f"epidemic_rl.parameters_{country}")
        self.base = Model.from_dict(copy.deepcopy(self.params.INITIAL_MODEL))
        self.population = self.base.population.astype(np.int64)
        self.data_dir = ROOT / "data" / country
        self.observed = pd.read_csv(self.data_dir / "pf_observations.csv", parse_dates=["date"]).set_index("date")
        self.gov = pd.read_csv(self.data_dir / "government_actions.csv", parse_dates=["date"]).set_index("date")
        self.scenario_q = pd.read_csv(self.data_dir / "scenario_q.csv", parse_dates=["date"]).set_index("date")
        if country == "england":
            self.home = np.asarray(self.params.HOUSEHOLD_SHARE) * self.base.contact_matrix
            self.nonhome = self.base.contact_matrix - self.home
            self.case_ages = AGES
            self.obs_columns = [f"cases_{a}" for a in AGES] + [f"hospital_additions_{a}" for a in AGES] + ["mv_beds_all_ages"]
            self.budget = 30.0
            self.q_sd = self.params.OBSERVATION_PARAMETERS["adaptive_process_selection"]["sigma_log_q_week"] / np.sqrt(7)
            self.rho_sd = self.params.OBSERVATION_PARAMETERS["adaptive_process_selection"]["sigma_logit_rho_week"] / np.sqrt(7)
            self.mapping = np.eye(3)
            self.weekday = np.ones((7, 7))
            # These observation refinements are frozen below in the prepared
            # source file; no residual-fitting procedure is called at runtime.
            self.hscale = np.asarray(getattr(self.params, "HOSPITAL_SCALE", [1., 1., 1.]))
            self.phi_case = np.asarray(getattr(self.params, "CASE_DISPERSION", [self.base.phi_case] * 3))
            self.weekday[:, :6] = np.asarray(getattr(self.params, "OBSERVATION_WEEKDAY", np.ones((7, 6))))
            op = self.params.OBSERVATION_PARAMETERS
            ar = [op["hospital_log_ar1"][a] for a in AGES]
            self.h_intercept = np.array([a["intercept"] for a in ar])
            self.h_phi = np.array([a["phi"] for a in ar])
            self.h_sigma = np.array([a["innovation_sd_used"] for a in ar])
            self.case_scale = np.array([op["case_age_calibration_scale"][a] for a in AGES])
            self.pi_mv = op["mv_delta"]["pi_mv_calibration_only"]
            self.c_mv = op["mv_delta"]["sigma_sqrt_previous_level_coefficient"]
        else:
            self.home = np.asarray(self.params.HOUSEHOLD_CONTACTS)
            self.nonhome = np.asarray(self.params.NONHOUSEHOLD_CONTACTS)
            self.case_ages = ("0_39", "40_59", "60_plus")
            self.obs_columns = [f"cases_{a}" for a in self.case_ages] + ["admissions_total", "icu_occupied_total"]
            self.mapping = np.asarray(self.params.OBSERVATION_MAPPING)
            self.weekday = np.asarray(self.params.WEEKDAY_FACTORS)
            self.budget = float(getattr(self.params, "GOVERNMENT_BUDGET", 61.713590902679336))
            self.q_sd, self.rho_sd = .1, 0.
            self.bootstrap_contacts = pd.read_csv(self.data_dir / "bootstrap_contacts.csv", parse_dates=["date"]).set_index("date")
        mortality = self.params.MORTALITY
        self.mortality = np.array([[mortality[a][key] for a in ("0-44", "45-64", "65+")]
                                  for key in ("p_death_given_I2_to_R", "p_death_given_H_to_R", "p_death_given_U_to_R")])
        self._models = {}
        self.counters = {phase: {"pf_updates": 0, "oracle_resets": 0} for phase in ("planning", "deployment", "evaluation", "historical", "bootstrap")}
        initial = self.contact_loss(np.array([1., 1., 1.]))
        if abs(initial) > 1e-10:
            raise ValueError("All-one actions must cost zero")
        raw_budget = sum(self.contact_loss(self.government_action(d.date())) for d in pd.date_range(START, CONTROL_END))
        self.cost_scale = self.budget / raw_budget

    def model(self, dt):
        if dt in self._models:
            return self._models[dt]
        values = copy.deepcopy(self.params.INITIAL_MODEL)
        if self.country == "belgium" and dt >= date(2020, 12, 1):
            values = copy.deepcopy(self.params.DECEMBER_MODEL)
        if self.country == "england" and date(2020, 12, 15) <= dt <= date(2021, 1, 31):
            p = self.params.LATE_CALIBRATION
            j = min((dt-date(2020, 12, 15)).days // 7, 6)
            values["p_hospital"] = np.clip(np.array(values["p_hospital"]) * p["hospital_admission_multiplier_by_age"] * p["hospital_weekly_multiplier"][j], 1e-9, .99)
            values["p_icu"] = np.clip(np.array(values["p_icu"]) * p["icu_branch_multiplier_by_age"], 1e-9, .99)
            values["p_exit_icu"] *= p["icu_exit_multiplier"]
            values["transmission_scale"] = p["q_weekly_multiplier"][j]
        model = Model.from_dict(values)
        self._models[dt] = model
        return model

    def contacts(self, action, dt=None):
        if dt is not None and self.country == "belgium" and dt < START:
            row = self.bootstrap_contacts.loc[pd.Timestamp(dt)]
            return row.to_numpy(float).reshape(3, 3)
        a = np.asarray(action, float)
        if a.shape != (3,) or np.any(a <= 0) or np.any(a > 1):
            raise ValueError("Three action multipliers in (0,1] required")
        scale = np.sqrt(a)
        return self.home + scale[:, None] * self.nonhome * scale[None, :]

    def government_action(self, dt):
        return self.gov.loc[pd.Timestamp(dt), [f"a_{a}" for a in AGES]].to_numpy(float)

    def contact_loss(self, action):
        baseline = self.home + self.nonhome
        return float((self.population[:, None] * (baseline - self.contacts(action))).sum() /
                     (self.population[:, None] * baseline).sum())

    def initial_belief(self, n, seed):
        rng = np.random.default_rng(seed)
        prior = self.params.INITIAL_PRIOR
        base = initial_cloud(np.asarray(prior["mean_x"]), np.asarray(prior["mean_case_history"]),
                             np.asarray(prior["mean_hospital_history"]), self.base, n,
                             .25 if self.country == "england" else .35, rng)
        q = np.full((n, 3), math.log(self.base.q))
        if self.country == "belgium":
            q += rng.normal(0, .08, size=(n, 1))
        b = Cloud(base.x, base.case_history, base.hospital_history, q,
                  np.zeros(n), np.zeros((n, 3)), base.x[:, :, U].sum(axis=1).astype(float),
                  np.full(n, float(self.observed.iloc[0]["mv_beds_all_ages"]) if self.country == "england" else 0.),
                  np.zeros((n, 3), dtype=np.int64), np.full(n, -math.log(n)))
        if self.country == "belgium":
            for timestamp in pd.date_range("2020-09-22", "2020-10-11"):
                dt = timestamp.date()
                self.predict(b, np.ones(3), dt, rng)
                self.assimilate(b, self.observed.loc[timestamp, self.obs_columns].to_numpy(float), dt, rng, "bootstrap")
            # Exact source q gauge change at the Oct11/Oct12 contact boundary.
            old = self.contacts(np.ones(3), date(2020, 10, 11))
            new = self.contacts(self.government_action(START))
            infectious = (b.x[:, :, I1] + self.base.xi*b.x[:, :, I2]) / self.population
            old_force, new_force = infectious @ old.T, infectious @ new.T
            b.logq += np.log(np.maximum(old_force, 1e-20) / np.maximum(new_force, 1e-20))
        self.assert_valid(b)
        return b

    def sample_truth(self, belief, rng):
        if belief.oracle_state is not None:
            # Planning starts from the corrected known current state, while its
            # filtering cloud retains the independently estimated uncertainty.
            return belief.oracle_state.copy()
        return belief.subset(np.array([rng.choice(belief.n, p=belief.weights)]))

    def predict(self, b, action, dt, rng, *, scenario=False):
        # A point correction is valid only at its current time, not tomorrow.
        b.oracle_state = None
        model = self.model(dt)
        if self.country == "england":
            b.logq += rng.normal(-.5*self.q_sd**2, self.q_sd, size=b.logq.shape)
            b.zrho += rng.normal(0, self.rho_sd, size=b.n)
        else:
            common = rng.normal(0, .1, (b.n, 1))
            ages = rng.normal(0, .08, (b.n, 3)); ages -= ages.mean(axis=1, keepdims=True)
            b.logq += common + ages
        if scenario and pd.Timestamp(dt) in self.scenario_q.index:
            b.logq[:] = np.log(self.scenario_q.loc[pd.Timestamp(dt)].to_numpy(float))
        # Preserve the source lognormal random walk; do not silently clip its
        # future transmission states to a new parameter-fitting bound.
        if not np.isfinite(b.logq).all():
            raise RuntimeError("Nonfinite transmission process")
        q = np.exp(b.logq) * model.transmission_scale
        x = b.x
        force = q * (((x[:, :, I1] + model.xi*x[:, :, I2])/self.population) @ self.contacts(action, dt).T)
        a = rng.binomial(x[:, :, S], np.clip(-np.expm1(-force), 0, 1))
        eb = rng.binomial(x[:, :, E1], model.pE)
        ec = rng.binomial(x[:, :, E2], model.pE)
        idi = rng.binomial(x[:, :, I1], model.pI)
        exiti = rng.binomial(x[:, :, I2], model.pI)
        ih = rng.binomial(exiti, model.p_hospital); ir = exiti - ih
        exith = rng.binomial(x[:, :, H], model.p_exit_hospital)
        hu = rng.binomial(exith, model.p_icu); hr = exith - hu
        ur = rng.binomial(x[:, :, U], model.p_exit_icu)
        b.x = x.copy()
        b.x[:, :, S] -= a; b.x[:, :, E1] += a-eb; b.x[:, :, E2] += eb-ec
        b.x[:, :, I1] += ec-idi; b.x[:, :, I2] += idi-ih-ir
        b.x[:, :, H] += ih-hu-hr; b.x[:, :, U] += hu-ur; b.x[:, :, R] += ir+hr+ur
        deaths = sum(rng.binomial(flow, p) for flow, p in zip((ir, hr, ur), self.mortality))
        b.deaths += deaths
        b.ch = shift_history(b.ch, a); b.hh = shift_history(b.hh, ih)
        self.assert_valid(b)
        return {"new_deaths": deaths, "new_infections": a, "admissions": ih,
                "icu": b.x[:, :, U].sum(axis=1), "deaths_total": deaths.sum(axis=1)}

    def observation_components(self, b, dt):
        model = self.model(dt)
        if self.country == "belgium":
            cases = ((b.ch @ model.delay_case) * model.rho_case) @ self.mapping.T
            mu = np.column_stack([cases, b.hh[:, :, 0].sum(axis=1), b.x[:, :, U].sum(axis=1)])
            return mu * self.weekday[dt.weekday()]
        rho = expit(logit(model.rho_case)[None, :] + b.zrho[:, None])
        cases = (b.ch @ model.delay_case) * rho * self.case_scale
        admissions = (b.hh @ model.delay_hospital) * self.hscale
        base = np.column_stack([cases, admissions]) * self.weekday[dt.weekday(), :6]
        hospital_logmean = np.log(base[:, 3:6]+.5) + self.h_intercept + self.h_phi*b.eH
        mv_mean = b.mv_prev + self.pi_mv*(b.x[:, :, U].sum(axis=1)-b.u_prev)
        return base, hospital_logmean, mv_mean

    def draw_observations(self, b, dt, rng):
        model = self.model(dt)
        if self.country == "belgium":
            mu = self.observation_components(b, dt)
            phi = np.array([model.phi_case]*3+[45., 45.])
            return rng.negative_binomial(phi, phi/(phi+np.maximum(mu, 1e-12)))
        base, hm, mv = self.observation_components(b, dt)
        op = self.params.OBSERVATION_PARAMETERS["robust_cases"]
        phi = np.broadcast_to(self.phi_case, (b.n, 3)).copy()
        outlier = rng.random(b.n) < op["common_day_outlier_probability"]
        phi[outlier] *= op["outlier_nb_size_factor"]
        cases = rng.negative_binomial(phi, phi/(phi+np.maximum(base[:, :3], 1e-12)))
        hosp = np.maximum(0, np.rint(np.exp(np.clip(rng.normal(hm, self.h_sigma), -30, 30))-.5))
        mv = np.maximum(0, np.rint(rng.normal(mv, self.c_mv*np.sqrt(np.maximum(b.mv_prev, 1)))))
        return np.column_stack([cases, hosp, mv]).astype(np.int64)

    def condition_observation_states(self, b, y, dt):
        if self.country == "england":
            base, _, _ = self.observation_components(b, dt)
            b.eH = np.log((y[3:6][None, :]+.5)/(base[:, 3:6]+.5))
            b.u_prev = b.x[:, :, U].sum(axis=1).astype(float)
            b.mv_prev[:] = float(y[6])

    def assimilate(self, b, y, dt, rng, phase="deployment"):
        """Weight the already propagated predictive particles, then resample."""
        b.oracle_state = None
        y = np.asarray(y, float)
        model = self.model(dt)
        if self.country == "belgium":
            mu = self.observation_components(b, dt)
            ll = nb_logpmf(y, np.maximum(mu, 1e-12), np.array([model.phi_case]*3+[45.,45.])).sum(axis=1)
        else:
            base, hm, mv = self.observation_components(b, dt)
            op = self.params.OBSERVATION_PARAMETERS["robust_cases"]
            eps, factor = op["common_day_outlier_probability"], op["outlier_nb_size_factor"]
            usual = nb_logpmf(y[:3], base[:, :3], self.phi_case).sum(axis=1)
            outlier = nb_logpmf(y[:3], base[:, :3], self.phi_case*factor).sum(axis=1)
            cases = np.logaddexp(np.log1p(-eps)+usual, np.log(eps)+outlier)
            lh = np.log(y[3:6]+.5)
            ll = cases + (norm.logpdf(lh, hm, self.h_sigma)-lh).sum(axis=1)
            ll += norm.logpdf(y[6], mv, self.c_mv*np.sqrt(np.maximum(b.mv_prev, 1)))
        unnormalized = b.logw + ll
        evidence = logsumexp(unnormalized)
        if not np.isfinite(evidence):
            raise RuntimeError("Particle weights collapsed")
        b.logw = unnormalized-evidence
        ess = 1/np.dot(b.weights, b.weights)
        self.condition_observation_states(b, y, dt)
        resampled = ess < .5*b.n
        if resampled:
            b.resample(systematic_resample(b.weights, rng))
        self.counters[phase]["pf_updates"] += 1
        self.assert_valid(b)
        return {"ess_fraction": ess/b.n, "resampled": bool(resampled), "logscore": float(evidence)}

    def oracle_gate(self, b, truth, *, enabled=True, phase="deployment"):
        """Correct the policy point estimate; leave PF particles and SD intact."""
        b.oracle_state = None
        if not enabled:
            return b, []
        triggers = []
        med, _ = self.state_statistics(b)
        true = self.state_matrix(truth)[0]
        keys = self.state_names()
        for g, age in enumerate(AGES):
            for key, floor, limit in (("E",80.,1.30),("I",40.,1.25),("H",5.,1.5),("U",3.,1.5)):
                j = g*12+keys.index(key)
                ratio = (true[j]+floor)/(med[j]+floor)
                if max(ratio, 1/ratio) > limit:
                    triggers.append(f"{age}_{key}")
            j = g*12+keys.index("q")
            ratio = true[j]/max(med[j], 1e-20)
            if max(ratio, 1/ratio) > 1.2:
                triggers.append(f"{age}_q")
            j = g*12+keys.index("S")
            if abs(true[j]-med[j])/self.population[g] > .0005:
                triggers.append(f"{age}_S")
        if triggers:
            self.correct_policy_point(b, truth)
            # Keep the existing diagnostic counter name for output compatibility.
            self.counters[phase]["oracle_resets"] += 1
        return b, triggers

    @staticmethod
    def correct_policy_point(belief, truth):
        belief.oracle_state = truth.copy()
        belief.oracle_state.oracle_state = None

    def step_with_filter(self, truth, belief, action, dt, rng_truth, rng_pf,
                         *, phase="deployment", oracle=True, scenario=False):
        metrics = self.predict(truth, action, dt, rng_truth, scenario=scenario)
        y = self.draw_observations(truth, dt, rng_truth)[0]
        self.condition_observation_states(truth, y, dt)
        self.predict(belief, action, dt, rng_pf)
        try:
            diagnostic = self.assimilate(belief, y, dt, rng_pf, phase)
        except RuntimeError as error:
            if not oracle or "Particle weights collapsed" not in str(error):
                raise
            # Failed likelihood update: retain the predictive cloud and its
            # uncertainty, condition observation buffers, and correct only the
            # current policy point. No replicated-truth particle collapse.
            self.condition_observation_states(belief, y, dt)
            self.correct_policy_point(belief, truth)
            self.counters[phase]["oracle_resets"] += 1
            self.counters[phase]["pf_updates"] += 1
            weights = belief.weights
            diagnostic = {"ess_fraction": float(1/np.dot(weights, weights)/belief.n),
                          "resampled": False, "logscore": None,
                          "observation_update_failed": True, "oracle_point_corrected": True}
            return truth, belief, y, metrics, diagnostic, ["numeric_filter_collapse"]
        belief, triggers = self.oracle_gate(belief, truth, enabled=oracle, phase=phase)
        diagnostic["oracle_point_corrected"] = bool(triggers)
        return truth, belief, y, metrics, diagnostic, triggers

    @staticmethod
    def state_names():
        return (*COMPARTMENTS, "D", "q", "E", "I")

    def state_matrix(self, b):
        states = np.concatenate([b.x.astype(float), b.deaths[:, :, None], np.exp(b.logq)[:, :, None],
                                 (b.x[:, :, E1]+b.x[:, :, E2])[:, :, None],
                                 (b.x[:, :, I1]+b.x[:, :, I2])[:, :, None]], axis=2)
        return states.reshape(b.n, -1)

    def state_statistics(self, b):
        """Actual weighted PF posterior statistics, unaffected by the oracle."""
        return weighted_summary(self.state_matrix(b), b.weights)

    def policy_statistics(self, b):
        median, std = self.state_statistics(b)
        if b.oracle_state is not None:
            median = self.state_matrix(b.oracle_state)[0].copy()
        return median, std

    def state_row(self, b):
        med, std = self.state_statistics(b)
        row = {f"{age}_{state}_{stat}": float(values[g*12+j])
                for g, age in enumerate(AGES) for j, state in enumerate(self.state_names())
                for stat, values in (("median", med), ("std", std))}
        center = med if b.oracle_state is None else self.state_matrix(b.oracle_state)[0]
        row.update({f"{age}_{state}_policy_estimate": float(center[g*12+j])
                    for g, age in enumerate(AGES) for j, state in enumerate(self.state_names())})
        row["oracle_point_corrected"] = b.oracle_state is not None
        return row

    def feature_names(self):
        return [f"{age}_{state}_{stat}" for age in AGES for state in self.state_names()
                for stat in ("median", "std")] + [f"previous_action_lag{lag}_{age}" for lag in (1,2,3) for age in AGES] + ["remaining_control_fraction", "remaining_budget_fraction"]

    def features(self, b, action_history, remaining_days, spent):
        med, std = self.policy_statistics(b)
        scales = np.repeat(self.population.astype(float), 12)
        for g in range(3):
            scales[g*12+9] = self.base.q
        values = np.column_stack([med/scales, std/scales]).ravel().tolist()
        history = list(action_history)
        for lag in (1,2,3):
            values.extend(history[-lag] if len(history)>=lag else np.ones(3))
        values.extend([max(remaining_days, 0)/96., (self.budget-spent)/self.budget])
        result = np.asarray(values, np.float32)
        if len(result) != len(self.feature_names()) or not np.isfinite(result).all():
            raise RuntimeError("Invalid median/SD policy state")
        return result

    def assert_valid(self, b):
        if b.x.shape != (b.n, 3, 8) or np.any(b.x < 0):
            raise RuntimeError("Invalid compartment values")
        if not np.array_equal(b.x.sum(axis=2), np.broadcast_to(self.population, (b.n, 3))):
            raise RuntimeError("Population conservation failed")
        if not np.isfinite(b.logq).all() or not np.isfinite(b.logw).all():
            raise RuntimeError("Nonfinite particle state")
