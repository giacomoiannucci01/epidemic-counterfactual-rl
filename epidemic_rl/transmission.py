"""Zero-drift daily log-transmission innovations shared by both countries."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TransmissionProcess:
    common_daily_sd: float
    age_daily_sd: float

    def __post_init__(self):
        if not np.isfinite([self.common_daily_sd, self.age_daily_sd]).all() or min(
                self.common_daily_sd, self.age_daily_sd) < 0:
            raise ValueError("Transmission innovation SDs must be finite and nonnegative")

    @property
    def covariance(self):
        # eps_g = common + age_g - mean_h(age_h), with independent raw draws.
        ones = np.ones((3, 3))
        return self.common_daily_sd**2 * ones + self.age_daily_sd**2 * (np.eye(3)-ones/3)

    def innovations(self, n, rng):
        # Draw order and shapes retain the original Belgium process exactly.
        common = rng.normal(0, self.common_daily_sd, (n, 1))
        ages = rng.normal(0, self.age_daily_sd, (n, 3))
        ages -= ages.mean(axis=1, keepdims=True)
        return common + ages

    def advance(self, logq, rng):
        if logq.ndim != 2 or logq.shape[1] != 3:
            raise ValueError("Expected one log-transmission value per particle and age")
        logq += self.innovations(len(logq), rng)
        if not np.isfinite(logq).all():
            raise RuntimeError("Nonfinite transmission process")

    def manifest(self):
        return {"law": "r[t+1] = r[t] + common + centered_age_innovations",
                "daily_log_drift": [0., 0., 0.],
                "common_daily_sd": self.common_daily_sd,
                "raw_age_daily_sd": self.age_daily_sd,
                "age_innovations_sum_to_zero": True,
                "daily_log_innovation_covariance": self.covariance.tolist(),
                "parameters_estimated_during_run": False,
                "innovation_parameter_source": "Original Belgium simulator; same fixed specification adopted for England"}
