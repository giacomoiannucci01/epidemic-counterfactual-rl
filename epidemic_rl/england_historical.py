"""Frozen England historical fits with the paper's correlated log-q process.

Keep the archived clinical/observation parameters, priors, likelihood, contact
calendar and Dec1--5 replay; change only fixed q to the shared stochastic law.
"""
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from legacy.england_fit import reproduce as archived
from .parameters_england import TRANSMISSION_PROCESS
from .transmission import TransmissionProcess


@dataclass
class Cloud(archived.Cloud):
    logq: np.ndarray

    def resample(self, indices):
        self.logq = self.logq[indices].copy()
        super().resample(indices)


def random_streams(seed, branch):
    # Independent streams keep forecast draws from changing PF trajectories.
    return [np.random.default_rng(s) for s in np.random.SeedSequence([seed, branch]).spawn(5)]


def make_cloud(model, prior, settings, rng):
    base = archived.make_cloud(model, prior, settings, rng)
    return Cloud(base.x, base.case_history, base.hospital_history, base.logw,
                 base.roots, np.full((len(base.logw), 3), np.log(model.q)))


def propagate(cloud, model, contacts, transition_rng, process_rng, process):
    process.advance(cloud.logq, process_rng)
    cloud.x, infections, admissions = archived.advance(
        cloud.x, model, transition_rng, contacts=contacts, q=np.exp(cloud.logq))
    cloud.case_history = archived.shift_history(cloud.case_history, infections)
    cloud.hospital_history = archived.shift_history(cloud.hospital_history, admissions)
    if np.any(cloud.x < 0) or not np.array_equal(
            cloud.x.sum(axis=2), np.broadcast_to(model.population, (len(cloud.x), 3))):
        raise RuntimeError("Particle population conservation failed")
    return archived.observation_means(cloud.case_history, cloud.hospital_history, model)


def predict_day(cloud, model, contacts, transition_rng, prediction_rng, process_rng, draws, process):
    """Construct a forecast without accepting target-day observations."""
    weights = np.exp(cloud.logw)
    mu = propagate(cloud, model, contacts, transition_rng, process_rng, process)
    mean, variance, quantiles = archived.predictive_summary(mu, weights, model.phi, prediction_rng, draws)
    return mu, mean, variance, quantiles


def replay_training_prefix(model, prior, settings, dates, y, contacts, seed, process):
    init_rng, transition_rng, resample_rng, _, process_rng = random_streams(seed, 1)
    cloud = make_cloud(model, prior, settings, init_rng)
    diagnostics = []
    for day, observations, contact in zip(dates, y, contacts):
        if not archived.CALIBRATION_START <= day <= archived.REPLAY_END:
            raise ValueError("Replay must use 1--5 December only")
        mu = propagate(cloud, model, contact, transition_rng, process_rng, process)
        diag = archived.assimilate_day(cloud, model, mu, observations, resample_rng,
                                       settings['ess_threshold_fraction'])
        diagnostics.append({'date': str(day), **diag})
    return cloud, diagnostics


def rolling_filter(initial_model, initial_prior, updated_model, updated_prior,
                   settings, dates, y, contacts, seed, state_callback=None):
    process = TransmissionProcess(**TRANSMISSION_PROCESS)
    init_rng, transition_rng, resample_rng, prediction_rng, process_rng = random_streams(seed, 0)
    cloud = make_cloud(initial_model, initial_prior, settings, init_rng)
    model = initial_model
    predictions, diagnostics, checks, replay_diagnostics = [], [], [], []
    for i, day in enumerate(dates):
        if day == archived.NEW_PARAMETERS_FROM:
            indices = [j for j, d in enumerate(dates) if archived.CALIBRATION_START <= d <= archived.REPLAY_END]
            cloud, replay_diagnostics = replay_training_prefix(updated_model, updated_prior, settings,
                [dates[j] for j in indices], y[indices], contacts[indices], seed, process)
            model = updated_model
            _, transition_rng, resample_rng, prediction_rng, process_rng = random_streams(seed, 2)
        # Freeze all forecasts before accessing this target day's observations.
        verify = day in (date(2020, 11, 2), archived.NEW_PARAMETERS_FROM)
        if verify:
            saved_cloud = deepcopy(cloud)
            saved_rngs = [deepcopy(rng) for rng in (transition_rng, prediction_rng, process_rng)]
        mu, mean, variance, quantiles = predict_day(cloud, model, contacts[i], transition_rng,
            prediction_rng, process_rng, settings['predictive_draws'], process)
        if verify:
            repeated = predict_day(saved_cloud, model, contacts[i], *saved_rngs,
                                   settings['predictive_draws'], process)
            if not all(np.array_equal(a, b) for a, b in zip((mu, mean, variance, quantiles), repeated)):
                raise RuntimeError("Pre-observation forecast reproducibility failed")
            checks.append({'target_date': str(day), 'forecast_observation_argument': False,
                           'identical_pre_observation_inputs_produce_identical_predictions': True})
        parameter_end = archived.CALIBRATION_END if day >= archived.NEW_PARAMETERS_FROM else date(2020, 11, 1)
        target = [{'date': str(day), 'forecast_origin': str(day-timedelta(days=1)),
                   'stream': stream, 'parameter_data_through': str(parameter_end),
                   'initial_calibration_overlap': day <= date(2020, 11, 1),
                   'retrospective_decjan_parameters': day >= archived.NEW_PARAMETERS_FROM,
                   'predictive_mean': float(mean[j]), 'predictive_median': float(quantiles[2, j]),
                   'lower_95': float(quantiles[0, j]), 'upper_95': float(quantiles[4, j]),
                   'predictive_sd': float(np.sqrt(variance[j]))}
                  for j, stream in enumerate(archived.OBS_COLUMNS)]
        observations = y[i]
        for j, row in enumerate(target):
            row['observed'] = int(observations[j])
            row['covered_95'] = bool(row['lower_95'] <= observations[j] <= row['upper_95'])
        predictions.extend(target)
        diag = archived.assimilate_day(cloud, model, mu, observations, resample_rng,
                                       settings['ess_threshold_fraction'])
        weights = np.exp(cloud.logw)
        diagnostics.append({'date': str(day), 'parameter_regime':
                            'recalibrated' if day >= archived.NEW_PARAMETERS_FROM else 'initial',
                            **diag, 'ess_after_resampling': float(1/(weights @ weights)),
                            'ancestors_after_resampling': int(len(np.unique(cloud.roots)))})
        if state_callback is not None:
            state_callback(day, cloud, model)
        if i == 0 or (i+1) % 14 == 0 or day in (archived.REPLAY_END, archived.NEW_PARAMETERS_FROM, archived.END):
            print(f"{day}: forecast before assimilation; ESS/N={diag['ess_fraction']:.4f}", flush=True)
    return pd.DataFrame(predictions), pd.DataFrame(diagnostics), checks, pd.DataFrame(replay_diagnostics)


def diagnostic_summary(diagnostics):
    rows = []
    for regime, frame in [('all_requested_dates', diagnostics), *list(diagnostics.groupby('parameter_regime', sort=False))]:
        rows.append({'evaluation_window': regime, 'days': len(frame),
                     'ess_fraction_mean': float(frame.ess_fraction.mean()),
                     'ess_fraction_median': float(frame.ess_fraction.median()),
                     'ess_fraction_min': float(frame.ess_fraction.min()),
                     'ess_fraction_p05': float(frame.ess_fraction.quantile(.05)),
                     'ess_median': float(frame.ess_before_resampling.median()),
                     'ess_min': float(frame.ess_before_resampling.min()),
                     'resampled_days': int(frame.resampled.sum()),
                     'days_ess_below_10_percent': int((frame.ess_fraction < .1).sum()),
                     'days_ess_below_1_percent': int((frame.ess_fraction < .01).sum()),
                     'maximum_weight_max': float(frame.maximum_weight.max())})
    return pd.DataFrame(rows)


def plot_diagnostics(diagnostics, output):
    from .plotting import plt, save, style
    fig, ax = plt.subplots(figsize=(11.5, 4.2))
    dt = pd.to_datetime(diagnostics.date)
    ax.plot(dt, diagnostics.ess_fraction, color='#0072B2', lw=1.5, label='ESS / particles, before resampling')
    selected = diagnostics.resampled.to_numpy(bool)
    ax.scatter(dt[selected], diagnostics.ess_fraction[selected], s=16, color='#D55E00', label='Systematic resampling')
    ax.axhline(.5, ls='--', color='#5D6871', lw=1, label='Resampling threshold')
    ax.axvline(pd.Timestamp(archived.NEW_PARAMETERS_FROM), color='#009E73', ls=':', lw=1.3)
    ax.set_title('England | Particle-filter effective sample size')
    ax.set_ylabel('ESS / particles'); ax.set_xlabel('Date'); style(ax); ax.set_ylim(0, 1.03)
    ax.set_xlim(dt.min(), dt.max()); ax.legend(frameon=False, fontsize=8.5, loc='upper left')
    fig.tight_layout(); save(fig, output, 'particle_filter_ess')
