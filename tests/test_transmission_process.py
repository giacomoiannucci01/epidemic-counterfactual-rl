"""Covariance, common fluctuations, PF ancestry and forecast timing."""
from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
import unittest

import numpy as np
import pandas as pd

from epidemic_rl.engine import Engine, START
from epidemic_rl.transmission import TransmissionProcess
from epidemic_rl import england_historical as current
from legacy.england_fit import reproduce as archived


class TransmissionTests(unittest.TestCase):
    def test_daily_innovation_covariance_and_zero_log_drift(self):
        process = TransmissionProcess(.10, .08)
        expected = np.full((3, 3), .01-.0064/3)
        np.fill_diagonal(expected, .01+2*.0064/3)
        np.testing.assert_allclose(process.covariance, expected, atol=1e-17, rtol=0)
        eps = process.innovations(200000, np.random.default_rng(19))
        np.testing.assert_allclose(eps.mean(axis=0), 0, atol=.0008, rtol=0)
        np.testing.assert_allclose(np.cov(eps, rowvar=False), expected, atol=.00015, rtol=0)
        # Removing the common shock leaves only a zero-sum age contrast.
        centered = eps-eps.mean(axis=1, keepdims=True)
        np.testing.assert_allclose(centered.sum(axis=1), 0, atol=3e-16, rtol=0)
        np.testing.assert_allclose(np.var(eps.mean(axis=1)), .01, atol=.00015, rtol=0)

    def test_both_country_prediction_paths_use_same_law_and_belgium_draws_are_unchanged(self):
        for country in ('england', 'belgium'):
            with self.subTest(country=country):
                engine = Engine(country); cloud = engine.initial_belief(16, 2026)
                r1, r2 = np.random.default_rng(31), np.random.default_rng(31)
                common = r1.normal(0, .1, (cloud.n, 1))
                ages = r1.normal(0, .08, (cloud.n, 3))
                ages -= ages.mean(axis=1, keepdims=True)
                expected = cloud.logq + (common + ages)
                engine.predict(cloud, engine.government_action(START), START, r2)
                np.testing.assert_array_equal(cloud.logq, expected)
                self.assertTrue(np.all(np.exp(cloud.logq) > 0))
                self.assertEqual(engine.model(START).xi, 1.)

    def test_historical_resampling_preserves_logq_with_all_ancestor_arrays(self):
        model = archived.Model.from_dict(archived.INITIAL_MODEL_PARAMETERS)
        cloud = current.make_cloud(model, archived.INITIAL_PRIOR,
            {**archived.SETTINGS, 'particles': 8}, np.random.default_rng(4))
        cloud.logq[:] = np.arange(24).reshape(8, 3)/100-3
        before = deepcopy(cloud); indices = np.array([5, 5, 0, 1, 1, 2, 7, 7])
        cloud.resample(indices)
        for key in ('x', 'case_history', 'hospital_history', 'roots', 'logq'):
            np.testing.assert_array_equal(getattr(cloud, key), getattr(before, key)[indices])
        np.testing.assert_allclose(np.exp(cloud.logw), 1/8, atol=1e-16, rtol=0)

    def test_zero_weight_particles_are_valid_without_resetting_weights(self):
        engine = Engine('england'); cloud = engine.initial_belief(8, 19)
        cloud.logw[:] = -np.inf
        cloud.logw[:2] = np.log([.6, .4])
        engine.assert_valid(cloud)
        np.testing.assert_array_equal(cloud.weights[2:], np.zeros(6))
        np.testing.assert_allclose(cloud.weights[:2], [.6, .4], atol=1e-16, rtol=0)
        for invalid in (-np.inf, np.inf, np.nan):
            broken = cloud.copy(); broken.logw[:] = invalid
            with self.assertRaisesRegex(RuntimeError, 'Nonfinite particle state'):
                engine.assert_valid(broken)

    def test_rolling_forecasts_reproducible_and_do_not_use_current_or_future_observations(self):
        observed = pd.read_csv(archived.DATA_PATH)
        dates = pd.to_datetime(observed.date).dt.date.tolist()
        y = observed[list(archived.OBS_COLUMNS)].to_numpy(np.int64)
        initial = archived.Model.from_dict(archived.INITIAL_MODEL_PARAMETERS)
        updated = archived.Model.from_dict(archived.UPDATED_MODEL_PARAMETERS)
        contacts, _ = archived.contact_calendar(initial, dates)
        settings = {**archived.SETTINGS, 'particles': 48, 'predictive_draws': 80}

        def run(values, draws=80):
            with redirect_stdout(StringIO()):
                return current.rolling_filter(initial, archived.INITIAL_PRIOR,
                    updated, archived.UPDATED_PRIOR, {**settings, 'predictive_draws': draws},
                    dates, values, contacts, 47)

        first = run(y); repeat = run(y)
        for a, b in zip(first, repeat):
            if isinstance(a, pd.DataFrame):
                pd.testing.assert_frame_equal(a, b, check_exact=True)
            else:
                self.assertEqual(a, b)
        # Include the Dec6 branch: its replay must stop at Dec5.
        changed = y.copy(); changed[np.array(dates) >= archived.NEW_PARAMETERS_FROM] += 100
        alternate = run(changed)
        for key in ('predictive_mean', 'predictive_median', 'lower_95', 'upper_95'):
            mask = first[0].date.le(str(archived.NEW_PARAMETERS_FROM))
            np.testing.assert_array_equal(first[0].loc[mask, key], alternate[0].loc[mask, key])
        self.assertEqual(first[3].date.tolist(), pd.date_range('2020-12-01', '2020-12-05').strftime('%Y-%m-%d').tolist())
        # Predictive MC sample size must never perturb filtering or its ESS.
        other_draws = run(y, draws=100)
        pd.testing.assert_frame_equal(first[1], other_draws[1], check_exact=True)
        diag = first[1]
        np.testing.assert_allclose(diag.loc[diag.resampled, 'ess_after_resampling'], 48, rtol=1e-14)
        self.assertTrue(np.isfinite(diag.ess_before_resampling).all())


if __name__ == '__main__':
    unittest.main()
