"""Reproduce England cases using a stochastic PF and fixed, already fitted parameters.

Run: python -m legacy.england_fit.reproduce
Check a second identical run: python -m legacy.england_fit.reproduce --verify-reproducibility
All model parameters and initial conditions are literal values below.
The only input data file is data/england/pf_observations.csv.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln, xlogy, logsumexp
from epidemic_rl.paths import ROOT


BASE = Path(__file__).resolve().parent
OUTPUT = Path.cwd() / "outputs/england/legacy"
DATA_PATH = ROOT / "data/england/pf_observations.csv"
AGES = ("0_44", "45_64", "65_plus")
OBS_COLUMNS = tuple("cases_" + age for age in AGES) + tuple("hospital_additions_" + age for age in AGES)
COMPARTMENTS = ("S", "E1", "E2", "I1", "I2", "H", "U", "R")
S, E1, E2, I1, I2, H, U, R = range(8)
START, END = date(2020, 10, 12), date(2021, 1, 31)
CALIBRATION_START, CALIBRATION_END = date(2020, 12, 1), date(2021, 1, 31)
NEW_PARAMETERS_FROM = date(2020, 12, 6)
REPLAY_END = date(2020, 12, 5)

# Numerical/scenario settings, fixed for reproduction (not fitted to surveillance).
# Compartments, contact rows/columns, reporting rates and histories all use age order 0-44, 45-64, 65+.
# The initial removed-population fraction used to construct both literal priors was 0.06.



# Fixed particle count, prediction draws, seed, ESS threshold and prior log SD.
SETTINGS = {'particles': 5000,
 'predictive_draws': 20000,
 'seed': 20261114,
 'ess_threshold_fraction': 0.5,
 'initial_active_log_sd': 0.25}



# User-supplied historical contact multipliers, not fitted by this script.
# The last historical period is retained through 31 January for this figure.
POLICY = (('2020-10-12', '2020-11-04', (0.878, 0.927, 0.868)),
 ('2020-11-05', '2020-12-02', (0.612, 0.607, 0.269)),
 ('2020-12-03', '2020-12-19', (0.753, 0.881, 0.892)),
 ('2020-12-20', '2021-01-04', (0.056, 0.785, 0.526)),
 ('2021-01-05', '2021-01-31', (0.059, 0.786, 0.248)))



# Fixed contact-reference input: Prem 2021 UK home/all household shares.
# These shares are anchored to the original model's total contact matrix.
HOUSEHOLD_SHARE = [[0.3455218440105344, 0.3780969175674851, 0.5792172657289467],
 [0.37809691756748515, 0.3474157757852531, 0.39222714240893874],
 [0.5792172657289466, 0.3922271424089388, 0.6012470186906546]]



# INITIAL FIT: observations from 12 Oct 2020 through 1 Nov 2020, inclusive.
# Applied to the rolling filter from 12 Oct 2020 through 5 Dec 2020.
# Previously estimated offline: q, rho_case, phi_case, phi_hospital and case-delay probabilities.
# Case-delay gamma mean = 12.0 days; shape = 3.0; maximum lag = 28 days.
# Population, contact matrix, clinical transitions, xi and transmission_scale are fixed reference values.
INITIAL_MODEL_PARAMETERS = {'population': [31613534, 14472585, 10464019],
 'contact_matrix': [[6.5518619776902955, 1.3099309167818296, 0.4296515982895252],
                    [2.8613786393608014, 1.3173313374169617, 0.6397911121402169],
                    [1.2980486188605207, 0.8848828784326386, 1.0088779541208432]],
 'q': 0.04342198703197486,
 'pE': 0.6666666666666666,
 'pI': 0.5,
 'p_hospital': [0.0020433674302426205, 0.01727817630936254, 0.05549757700737431],
 'p_icu': [0.12880760548124162, 0.3693652809007758, 0.1704456670934679],
 'p_exit_hospital': [0.12668747084272525, 0.17579545276105535, 0.1523014778430458],
 'p_exit_icu': 0.13545451491570937,
 'rho_case': [0.16228528764789812, 0.3396680439612737, 0.2703832575522013],
 'phi_case': 20.585844287518032,
 'phi_hospital': 8.204199907927306,
 'delay_case': [0.0002964775408880204,
                0.006355736706534974,
                0.019004716651602575,
                0.033146790295592206,
                0.04586564616868336,
                0.05582416219125151,
                0.06262837686640052,
                0.06641328624277668,
                0.06757953283806856,
                0.066631446810263,
                0.06408210803170755,
                0.06040185253613878,
                0.05599420721549253,
                0.051188515344150365,
                0.04624217203703207,
                0.04134789807174466,
                0.03664318338644257,
                0.03222017145760314,
                0.028135005780815958,
                0.024416142268461782,
                0.0210714326231638,
                0.018093963740918984,
                0.01546673855541325,
                0.01316633299128045,
                0.01116568119022976,
                0.009436139831404455,
                0.00794897066158562,
                0.006676363709150368,
                0.0325569482552025],
 'delay_hospital': [1.0],
 'xi': 1.0,
 'transmission_scale': 1.0}



# INITIAL STATE FIT: 12 Oct-1 Nov 2020, same initial fit as above.
# Literal end-of-day 11 Oct state and lag buffers, in compartment/age order above.
# Fitted incidence scales = [61312.60166005304, 12396.216442966228, 7898.2460272217795].
# Fitted prehistory growth = [0.008919647342672603] * 3.
# These arrays freeze the original deterministic state reconstruction; no state parameters are fitted at run time.
INITIAL_PRIOR = {'mean_x': [[29302940.23186005,
             90749.29379052541,
             89545.858442395,
             117292.69926909197,
             115227.91959172452,
             867.8885225045176,
             98.06852370947968,
             1896812.04],
            [13519187.415841235,
             18347.743488539985,
             18104.432249958823,
             23714.304204232216,
             23296.845882538437,
             1089.353504504432,
             489.80482899168095,
             868355.1],
            [9779985.90567418,
             11690.259909834976,
             11535.23421854088,
             15109.562650116768,
             14843.579190888011,
             2554.1902807849015,
             459.12807565426755,
             627841.14]],
 'mean_case_history': [[60768.14665862376,
                        60228.52640960354,
                        59693.69798046205,
                        59163.618819909214,
                        58638.24675450978,
                        58117.539985327974,
                        57601.45708460198,
                        57089.956992447864,
                        56582.99901359289,
                        56080.54281413773,
                        55582.54841834748,
                        55088.976205471205,
                        54599.78690658961,
                        54114.94160149088,
                        53634.401715574066,
                        53158.12901678013,
                        52686.08561255013,
                        52218.23394681054,
                        51754.53679698516,
                        51294.95727103376,
                        50839.458804516864,
                        50388.00515768669,
                        49940.560412603874,
                        49497.08897027983,
                        49057.555547844466,
                        48621.92517573908,
                        48190.163194934095,
                        47762.23525417165,
                        47338.1073072325],
                       [12286.138223180455,
                        12177.037496368202,
                        12068.905582406229,
                        11961.73387825067,
                        11855.513857252561,
                        11750.23706847945,
                        11645.895136043047,
                        11542.479758432824,
                        11439.982707855546,
                        11338.39582958066,
                        11237.711041291499,
                        11137.920332442252,
                        11039.015763620635,
                        10940.989465916233,
                        10843.83364029443,
                        10747.540556975933,
                        10652.102554821768,
                        10557.512040723768,
                        10463.761489000452,
                        10370.843440798279,
                        10278.750503498222,
                        10187.475350127599,
                        10097.010718777135,
                        10007.34941202321,
                        9918.484296355213,
                        9830.40830160801,
                        9743.114420399415,
                        9656.595707572713,
                        9570.845279644062],
                       [7828.1097186064235,
                        7758.596219380551,
                        7689.699999005935,
                        7621.415576055452,
                        7553.737517776971,
                        7486.6604396611165,
                        7420.179005012879,
                        7354.287924527023,
                        7288.981955867264,
                        7224.255903249189,
                        7160.104617026871,
                        7096.522993283166,
                        7033.505973423637,
                        6971.048543774095,
                        6909.1457351817,
                        6847.792622619616,
                        6786.984324795176,
                        6726.716003761518,
                        6666.982864532678,
                        6607.780154702097,
                        6549.103164064515,
                        6490.9472242412285,
                        6433.307708308664,
                        6376.180030430263,
                        6319.55964549163,
                        6263.442048738918,
                        6207.822775420425,
                        6152.697400431382,
                        6098.061537961884]],
 'mean_hospital_history': [[116.6810794826936], [199.47628833679056], [408.2337505648998]]}



# UPDATED FIT: observations from 1 Dec 2020 through 31 Jan 2021, inclusive.
# Applied from 6 Dec 2020 through 31 Jan 2021.
# Previously estimated offline: q, rho_case, phi_case, phi_hospital and case-delay probabilities.
# Case-delay gamma mean = 8.84653744447813 days; shape = 3.0; maximum lag = 28 days.
# Fixed demographic, contact and clinical reference inputs are unchanged.
UPDATED_MODEL_PARAMETERS = {'population': [31613534, 14472585, 10464019],
 'contact_matrix': [[6.5518619776902955, 1.3099309167818296, 0.4296515982895252],
                    [2.8613786393608014, 1.3173313374169617, 0.6397911121402169],
                    [1.2980486188605207, 0.8848828784326386, 1.0088779541208432]],
 'q': 0.08487967042895522,
 'pE': 0.6666666666666666,
 'pI': 0.5,
 'p_hospital': [0.0020433674302426205, 0.01727817630936254, 0.05549757700737431],
 'p_icu': [0.12880760548124162, 0.3693652809007758, 0.1704456670934679],
 'p_exit_hospital': [0.12668747084272525, 0.17579545276105535, 0.1523014778430458],
 'p_exit_icu': 0.13545451491570937,
 'rho_case': [0.1452851209011756, 0.187470570468146, 0.15278919049904155],
 'phi_case': 13.454836514381405,
 'phi_hospital': 5.359363566504685,
 'delay_case': [0.0007158307500889365,
                0.01433801098362429,
                0.03948406428218058,
                0.06314769946406074,
                0.08002709351613117,
                0.08916449914775634,
                0.09154904187095503,
                0.08883600154786947,
                0.08271078166063511,
                0.0746128157129714,
                0.06565083568783558,
                0.05661218900287546,
                0.04801193556430816,
                0.04015295761462079,
                0.03318304277919537,
                0.02714311340654052,
                0.02200512390327991,
                0.01770026229256727,
                0.01413895978095614,
                0.011224419055535173,
                0.008861257921509935,
                0.006960619608578744,
                0.005442825416445585,
                0.004238388427786344,
                0.0032879888989046924,
                0.002541837316218931,
                0.0019587171000989834,
                0.0015048995467137338,
                0.004794787739755613],
 'delay_hospital': [1.0],
 'xi': 1.0,
 'transmission_scale': 1.0}



# UPDATED STATE FIT: 1 Dec 2020-31 Jan 2021, same updated fit as above.
# Literal end-of-day 30 Nov state and lag buffers.
# Fitted incidence scales = [23124.271907767452, 10797.348483012998, 9516.510316998132].
# Fitted prehistory growth = [-0.06, -0.06, -0.06].
# Assimilate 1-5 Dec into this prior before forecasting 6 Dec.
UPDATED_PRIOR = {'mean_x': [[29501810.597767398,
             38006.38740786315,
             41644.13592661413,
             62845.14752498753,
             71129.68872251303,
             1061.649622064282,
             224.3530285587118,
             1896812.04],
            [13499989.439341154,
             17746.211040065336,
             19444.774290280584,
             29344.100475903986,
             33212.37700752157,
             2440.6659248131955,
             2052.331920260698,
             868355.1],
            [9736724.839416265,
             15641.062314150802,
             17138.13307371534,
             25863.14921309685,
             29272.550473050254,
             8635.189264594306,
             2902.9362451263414,
             627841.14]],
 'mean_case_history': [[24554.197023919667,
                        26072.543771073208,
                        27684.7798375291,
                        29396.711014550503,
                        31214.502103482242,
                        33144.69911569444,
                        35194.252845294024,
                        37370.543899491204,
                        39681.4092767575,
                        42135.170588485176,
                        44740.66402577671,
                        47507.272179276944,
                        50444.95782663387,
                        53564.29980925922,
                        56876.53112758407,
                        60393.579391993524,
                        64128.1097751074,
                        68093.57062008177,
                        72304.24186917018,
                        76775.28648694007,
                        81522.80506332306,
                        86563.8937931295,
                        91916.70604081592,
                        97600.51771220496,
                        103635.79666856694,
                        110044.27643302821,
                        116849.03445472957,
                        124074.57521256979,
                        131746.91845779758],
                       [11465.0192250493,
                        12173.976420002504,
                        12926.773079540098,
                        13726.120084754382,
                        14574.895948262492,
                        15476.157179960997,
                        16433.149293762955,
                        17449.318494953575,
                        18528.32409025158,
                        19674.051665265924,
                        20890.627076800953,
                        22182.431310397304,
                        23554.11625661193,
                        25010.62146284872,
                        26557.19192106454,
                        28199.3969554055,
                        29943.1502777895,
                        31794.731283656733,
                        33760.80766457577,
                        35848.459419135244,
                        38065.20434858604,
                        40419.02512904599,
                        42918.39805775631,
                        45572.32357690703,
                        48390.35868495063,
                        51382.65135211924,
                        54559.977064078535,
                        57933.77762531518,
                        61516.202361991505],
                       [10104.978450164583,
                        10729.835420438041,
                        11393.331387838005,
                        12097.855654508758,
                        12845.945268787831,
                        13640.29416132037,
                        14483.762846119254,
                        15379.38872150526,
                        16330.397008021799,
                        17340.212362712537,
                        18412.47121158583,
                        19551.034844676153,
                        20760.00332085892,
                        22043.73023249103,
                        23406.83838304581,
                        24854.236434198738,
                        26391.13658231144,
                        28023.07332796849,
                        29755.92340615743,
                        31595.92694886244,
                        33549.70995627956,
                        35624.30815757435,
                        37827.19234710644,
                        40166.29528735855,
                        42650.04027544996,
                        45287.37147610429,
                        48087.78613030339,
                        51061.3687556132,
                        54218.82746133945]],
 'mean_hospital_history': [[77.16583289971427], [304.6670824420257], [862.5062667721643]]}



# Original stochastic transitions and observation likelihood.
@dataclass(frozen=True)
class Model:
    population: np.ndarray
    contact_matrix: np.ndarray
    q: float
    pE: float
    pI: float
    p_hospital: np.ndarray  # eventual probability after infection
    p_icu: np.ndarray      # eventual probability after hospitalisation
    p_exit_hospital: np.ndarray
    p_exit_icu: float
    rho_case: np.ndarray
    phi_case: float
    phi_hospital: float
    delay_case: np.ndarray
    delay_hospital: np.ndarray
    xi: float = 1.0
    transmission_scale: float = 1.0  # V18 calendar multiplier in infection hazard; baseline 1

    def __post_init__(self):
        for name in ('population', 'contact_matrix', 'p_hospital', 'p_icu',
                     'p_exit_hospital', 'rho_case', 'delay_case', 'delay_hospital'):
            a = np.array(getattr(self, name), copy=True)
            a.setflags(write=False)
            object.__setattr__(self, name, a)
        if self.population.shape != (3,) or self.contact_matrix.shape != (3, 3):
            raise ValueError('Exactly three age groups are required.')
        if np.any(self.population <= 0) or not np.isfinite(self.population).all():
            raise ValueError('Populations must be finite and positive.')
        if np.any(self.contact_matrix < 0) or not np.isfinite(self.contact_matrix).all():
            raise ValueError('Contacts must be finite and nonnegative.')
        if not np.isfinite(self.transmission_scale) or self.transmission_scale <= 0:
            raise ValueError('Positive finite transmission_scale required')
        if self.q <= 0 or not np.isfinite(self.q):
            raise ValueError('q must be a finite positive fixed constant.')
        for p in (self.pE, self.pI, self.p_hospital, self.p_icu,
                  self.p_exit_hospital, self.p_exit_icu, self.rho_case):
            if not np.isfinite(p).all() or np.any(np.asarray(p) <= 0) or np.any(np.asarray(p) > 1):
                raise ValueError('Transition/reporting probabilities must lie in (0,1].')
        if not np.isfinite([self.phi_case,self.phi_hospital]).all() or self.phi_case <= 0 or self.phi_hospital <= 0:
            raise ValueError('NB sizes must be positive.')
        for p in (self.delay_case, self.delay_hospital):
            if np.any(p < 0) or not np.isclose(p.sum(), 1):
                raise ValueError('Delay probabilities must sum to one.')

    @property
    def phi(self):
        return np.array([self.phi_case] * 3 + [self.phi_hospital] * 3)

    def to_dict(self):
        return {k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


def shift_history(history: np.ndarray, new_value: np.ndarray) -> np.ndarray:
    out = np.empty_like(history)
    out[..., 1:] = history[..., :-1]
    out[..., 0] = new_value
    return out


def advance(x: np.ndarray, model: Model, rng=None, contacts=None):
    """One transition; rng=None gives the deterministic mean-field map.

    Multinomial exits are sampled through two conditional binomials. This is
    EXACT multinomial sampling, not independent binomials from the same stock.
    New arrivals cannot move on again during this same day.
    """
    pop = model.population
    infectious = x[..., I1] + model.xi * x[..., I2]
    C = model.contact_matrix if contacts is None else contacts
    force = model.q * ((infectious / pop) @ C.T)
    p_infection = -np.expm1(-force)
    draw = (lambda n, p: n * p) if rng is None else rng.binomial
    a = draw(x[..., S], p_infection)
    b = draw(x[..., E1], model.pE)
    c = draw(x[..., E2], model.pE)
    d = draw(x[..., I1], model.pI)
    # A total exit then a branch split has probabilities
    # pI*p_hospital, pI*(1-p_hospital), 1-pI.
    exit_i = draw(x[..., I2], model.pI)
    ih = draw(exit_i, model.p_hospital)
    ir = exit_i - ih
    exit_h = draw(x[..., H], model.p_exit_hospital)
    hu = draw(exit_h, model.p_icu)
    hr = exit_h - hu
    ur = draw(x[..., U], model.p_exit_icu)
    out = x.copy()
    out[..., S] -= a
    out[..., E1] += a - b
    out[..., E2] += b - c
    out[..., I1] += c - d
    out[..., I2] += d - ih - ir
    out[..., H] += ih - hu - hr
    out[..., U] += hu - ur
    out[..., R] += ir + hr + ur
    return out, a, ih


def observation_means(case_history, hospital_history, model: Model):
    cases = (case_history @ model.delay_case) * model.rho_case
    hospital = hospital_history @ model.delay_hospital
    return np.concatenate((cases, hospital), axis=-1)


def nb_logpmf(y, mu, phi):
    """NB2 with Var = mu + mu**2/phi. Zero means are handled exactly.

    A positive observation with mean zero has log likelihood -inf. Do not
    silently clip the mean or reset a failed particle filter to uniform weights.
    """
    y, mu, phi = np.broadcast_arrays(np.asarray(y), np.asarray(mu), np.asarray(phi))
    if np.any(mu < 0) or np.any(phi <= 0) or np.any(y < 0):
        raise ValueError('Negative mean/count or nonpositive NB size.')
    out = (gammaln(y + phi) - gammaln(phi) - gammaln(y + 1) +
           xlogy(phi, phi / (phi + mu)) + xlogy(y, mu / (phi + mu)))
    return out


# Original bootstrap-particle-filter primitives.
@dataclass
class Cloud:
    x: np.ndarray
    case_history: np.ndarray
    hospital_history: np.ndarray
    logw: np.ndarray
    roots: np.ndarray

    def resample(self, indices):
        # EVERY part of each particle must follow the same ancestor index.
        self.x = self.x[indices].copy()
        self.case_history = self.case_history[indices].copy()
        self.hospital_history = self.hospital_history[indices].copy()
        self.roots = self.roots[indices].copy()
        self.logw = np.full(len(indices), -np.log(len(indices)))


def initial_cloud(mean_x, mean_ch, mean_hh, model: Model, n: int, log_sd: float, rng):
    """Explicit empirical-Bayes starting prior after the calibration window.

    Shared age multipliers keep active compartments and their prehistory on
    comparable scales. This is an assumed distribution, not a recovered
    posterior from the preceding deterministic calibration. R is scenario-fixed.
    """
    if n < 2 or log_sd < 0:
        raise ValueError('Need >=2 particles and nonnegative initial log SD.')
    scales = rng.lognormal(-0.5 * log_sd**2, log_sd, size=(n, 3))
    # Prior is truncated if active stock plus removed would exceed population.
    max_scale = (model.population - mean_x[:, R] - 1) / mean_x[:, 1:7].sum(axis=-1)
    if np.any(max_scale <= 0):
        raise ValueError('Invalid starting state.')
    bad = scales >= max_scale
    for _ in range(1000):
        if not bad.any():
            break
        scales[bad] = rng.lognormal(-.5*log_sd**2, log_sd, size=bad.sum())
        bad = scales >= max_scale
    if bad.any():
        raise RuntimeError('Cannot sample a population-conserving initial prior.')
    x = np.zeros((n, 3, 8), np.int64)
    x[:, :, 1:7] = np.rint(mean_x[None, :, 1:7] * scales[:, :, None]).astype(np.int64)
    x[:, :, R] = np.rint(mean_x[None, :, R]).astype(np.int64)
    x[:, :, 0] = model.population - x[:, :, 1:].sum(axis=-1)
    if np.any(x < 0):
        raise RuntimeError('Rounding caused an invalid initial particle.')
    ch = np.rint(mean_ch[None, :, :] * scales[:, :, None]).astype(np.int64)
    hh = np.rint(mean_hh[None, :, :] * scales[:, :, None]).astype(np.int64)
    return Cloud(x, ch, hh, np.full(n, -np.log(n)), np.arange(n))


def systematic_resample(weights, rng):
    n = len(weights)
    if np.any(weights < 0) or not np.isclose(np.sum(weights), 1):
        raise ValueError('Resampling weights must be nonnegative and normalised.')
    positions = (rng.random() + np.arange(n)) / n
    cdf = np.cumsum(weights)
    cdf[-1] = 1.0
    return np.searchsorted(cdf, positions, side='right')


def update_log_weights(logw, loglik):
    increment = float(logsumexp(logw + loglik))
    if not np.isfinite(increment):
        raise RuntimeError('ALL particles have zero likelihood. This run has failed; '
                           'do not interpret it as a posterior or silently reset weights.')
    new = logw + loglik - increment
    return new, increment


def predictive_summary(mu, weights, phi, rng, draws):
    mean = weights @ mu
    var = weights @ (mu + mu**2 / phi + (mu - mean)**2)
    idx = rng.choice(len(weights), size=draws, p=weights)
    draws_y = rng.negative_binomial(phi, phi / (phi + mu[idx]))
    quant = np.quantile(draws_y, [.025, .25, .5, .75, .975], axis=0)
    return mean, var, quant


def checksum(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False,
                               default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)), encoding="utf-8")


def random_streams(seed, branch):
    return [np.random.default_rng(child) for child in np.random.SeedSequence([seed, branch]).spawn(4)]


def make_cloud(model, prior, settings, rng):
    """Sample from the literal initial-state arrays above; do not estimate them."""
    return initial_cloud(np.asarray(prior["mean_x"], dtype=float),
                         np.asarray(prior["mean_case_history"], dtype=float),
                         np.asarray(prior["mean_hospital_history"], dtype=float),
                         model, settings["particles"], settings["initial_active_log_sd"], rng)



def contact_calendar(model, dates):
    share = np.asarray(HOUSEHOLD_SHARE, dtype=float)
    home = share * model.contact_matrix
    outside = model.contact_matrix - home
    matrices, rows = [], []
    for day in dates:
        matches = [entry for entry in POLICY if date.fromisoformat(entry[0]) <= day <= date.fromisoformat(entry[1])]
        if len(matches) != 1:
            raise ValueError(f"Policy is undefined or overlapping on {day}")
        action = np.asarray(matches[0][2])
        # Original belief_rl.core.action_contact_matrix: D(a) = diag(sqrt(a)).
        matrix = home + np.sqrt(np.outer(action, action)) * outside
        mass = model.population[:, None] * matrix
        if not np.allclose(mass, mass.T, rtol=1e-10, atol=1e-6):
            raise ValueError("Contact matrix does not preserve population reciprocity")
        matrices.append(matrix)
        rows.append({"date": str(day), **{f"a_{age}": float(a) for age, a in zip(AGES, action)},
                     **{f"C_{g+1}{h+1}": float(matrix[g,h]) for g in range(3) for h in range(3)}})
    return np.asarray(matrices), rows


def predict_day(cloud, model, contacts, transition_rng, prediction_rng, draws):
    """No observations are accepted here: target-day outcomes cannot enter."""
    weights = np.exp(cloud.logw)
    cloud.x, infections, admissions = advance(cloud.x, model, transition_rng, contacts=contacts)
    cloud.case_history = shift_history(cloud.case_history, infections)
    cloud.hospital_history = shift_history(cloud.hospital_history, admissions)
    if np.any(cloud.x < 0) or not np.array_equal(cloud.x.sum(axis=2),
                                               np.broadcast_to(model.population, (cloud.x.shape[0], 3))):
        raise RuntimeError("Particle population conservation failed")
    mu = observation_means(cloud.case_history, cloud.hospital_history, model)
    mean, variance, quantiles = predictive_summary(mu, weights, model.phi, prediction_rng, draws)
    return mu, mean, variance, quantiles


def assimilate_day(cloud, model, mu, observations, resampling_rng, ess_fraction):
    loglik = nb_logpmf(observations, mu, model.phi).sum(axis=1)
    cloud.logw, logscore = update_log_weights(cloud.logw, loglik)
    weights = np.exp(cloud.logw)
    ess = float(1 / (weights @ weights))
    diagnostics = {"ess_before_resampling": ess, "ess_fraction": ess / len(weights),
                   "maximum_weight": float(weights.max()), "joint_predictive_logscore": logscore,
                   "initial_ancestors_present": int(len(np.unique(cloud.roots))),
                   "resampled": ess < ess_fraction * len(weights)}
    if diagnostics["resampled"]:
        cloud.resample(systematic_resample(weights, resampling_rng))
    return diagnostics


def replay_training_prefix(model, prior, settings, dates, y, contacts, seed):
    rng_init, rng_transition, rng_resample, _ = random_streams(seed, 1)
    cloud = make_cloud(model, prior, settings, rng_init)
    for day, yt, contact in zip(dates, y, contacts):
        if not CALIBRATION_START <= day <= REPLAY_END:
            raise ValueError("State replay must use 1-5 December only")
        cloud.x, infections, admissions = advance(cloud.x, model, rng_transition, contacts=contact)
        cloud.case_history = shift_history(cloud.case_history, infections)
        cloud.hospital_history = shift_history(cloud.hospital_history, admissions)
        mu = observation_means(cloud.case_history, cloud.hospital_history, model)
        assimilate_day(cloud, model, mu, yt, rng_resample, settings["ess_threshold_fraction"])
    print("Rebuilt 5-December filtering distribution; updated parameters apply from 6 December", flush=True)
    return cloud


def rolling_filter(initial_model, initial_prior, updated_model, updated_prior,
                   settings, dates, y, contacts, seed, state_callback=None):
    rng_init, rng_transition, rng_resample, rng_predict = random_streams(seed, 0)
    cloud = make_cloud(initial_model, initial_prior, settings, rng_init)
    model = initial_model
    prediction_rows, diagnostics_rows, causal_checks = [], [], []
    for i, day in enumerate(dates):
        if day == NEW_PARAMETERS_FROM:
            replay_indices = [j for j, d in enumerate(dates) if CALIBRATION_START <= d <= REPLAY_END]
            replay_dates = [dates[j] for j in replay_indices]
            cloud = replay_training_prefix(updated_model, updated_prior, settings,
                                           replay_dates, y[replay_indices], contacts[replay_indices], seed)
            model = updated_model
            _, rng_transition, rng_resample, rng_predict = random_streams(seed, 2)
        params_end = CALIBRATION_END if day >= NEW_PARAMETERS_FROM else date(2020, 11, 1)
        # Causal check: two identical pre-observation inputs must forecast
        # identically even if target/future observation arrays are changed.
        if day in (date(2020, 11, 2), NEW_PARAMETERS_FROM):
            saved_cloud = deepcopy(cloud)
            tr_state, pr_state = deepcopy(rng_transition.bit_generator.state), deepcopy(rng_predict.bit_generator.state)
        mu, mean, variance, quantiles = predict_day(cloud, model, contacts[i], rng_transition,
                                                   rng_predict, settings["predictive_draws"])
        if day in (date(2020, 11, 2), NEW_PARAMETERS_FROM):
            test_tr, test_pr = np.random.default_rng(), np.random.default_rng()
            test_tr.bit_generator.state, test_pr.bit_generator.state = tr_state, pr_state
            repeated = predict_day(saved_cloud, model, contacts[i], test_tr, test_pr, settings["predictive_draws"])
            if not np.array_equal(quantiles, repeated[3]):
                raise RuntimeError("Causal forecast reproducibility check failed")
            causal_checks.append({"target_date": str(day), "forecast_observation_argument": False,
                                  "identical_pre_observation_inputs_produce_identical_predictions": True})
        # Freeze numerical predictions BEFORE reading or assimilating y[i].
        target_forecasts = []
        for j, stream in enumerate(OBS_COLUMNS):
            target_forecasts.append({"date": str(day), "forecast_origin": str(day - timedelta(days=1)),
                                    "stream": stream, "parameter_data_through": str(params_end),
                                    "initial_calibration_overlap": day <= date(2020, 11, 1),
                                    "retrospective_decjan_parameters": day >= NEW_PARAMETERS_FROM,
                                    "predictive_mean": float(mean[j]), "predictive_median": float(quantiles[2,j]),
                                    "lower_95": float(quantiles[0,j]), "upper_95": float(quantiles[4,j]),
                                    "predictive_sd": float(np.sqrt(variance[j]))})
        observations = y[i]
        for j, forecast in enumerate(target_forecasts):
            forecast["observed"] = int(observations[j])
            forecast["covered_95"] = bool(forecast["lower_95"] <= observations[j] <= forecast["upper_95"])
        prediction_rows.extend(target_forecasts)
        diag = assimilate_day(cloud, model, mu, observations, rng_resample, settings["ess_threshold_fraction"])
        diagnostics_rows.append({"date": str(day), "parameter_regime": "recalibrated" if day >= NEW_PARAMETERS_FROM else "initial", **diag})
        if state_callback is not None:
            state_callback(day, cloud, model)
        if i == 0 or (i + 1) % 14 == 0 or day in (REPLAY_END, NEW_PARAMETERS_FROM, END):
            print(f"{day}: forecast recorded before assimilation; ESS/N={diag['ess_fraction']:.4f}", flush=True)
    return pd.DataFrame(prediction_rows), pd.DataFrame(diagnostics_rows), causal_checks


def score_predictions(predictions, end_date=END):
    windows = [("all_requested_dates", START, end_date),
               ("after_initial_calibration", date(2020, 11, 2), end_date),
               ("initial_calibration_overlap", START, date(2020, 11, 1)),
               ("before_december_recalibration", date(2020, 11, 2), REPLAY_END),
               ("after_december_recalibration", NEW_PARAMETERS_FROM, end_date)]
    results = []
    for name, first, last in windows:
        for age in AGES:
            selected = predictions.loc[predictions.stream.eq(f"cases_{age}") &
                                       predictions.date.between(str(first), str(last))]
            actual = selected.observed.to_numpy()
            point = selected.predictive_mean.to_numpy()
            error = point - actual
            results.append({"evaluation_window": name, "age_group": age, "start_date": str(first), "end_date": str(last),
                            "days": len(selected), "point_forecast": "predictive_mean",
                            "wape_percent": float(100 * np.abs(error).sum() / actual.sum()),
                            "coverage_95_percent": float(100 * selected.covered_95.mean()),
                            "covered_days": int(selected.covered_95.sum()),
                            "mean_interval_width": float((selected.upper_95 - selected.lower_95).mean()),
                            "rmse_cases_per_day": float(np.sqrt(np.square(error).mean())),
                            "bias_percent": float(100 * error.sum() / actual.sum())})
    return pd.DataFrame(results)


def draw_plot(predictions, scores, end_date=END, filename_stem="cases_model_fit_by_age"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import StrMethodFormatter
    colors, labels = ("#0072B2", "#D55E00", "#009E73"), ("0–44", "45–64", "65+")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42,
                         "axes.labelcolor": "#26343F", "text.color": "#26343F"})
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 10), sharex=True)
    for ax, age, label, color in zip(axes, AGES, labels, colors):
        selected = predictions.loc[predictions.stream.eq(f"cases_{age}") & predictions.date.le(str(end_date))]
        dates = pd.to_datetime(selected.date)
        metrics = scores.loc[scores.evaluation_window.eq("all_requested_dates") & scores.age_group.eq(age)].iloc[0]
        ax.fill_between(dates, selected.lower_95.to_numpy(), selected.upper_95.to_numpy(),
                        color=color, alpha=0.18, linewidth=0)
        ax.plot(dates, selected.predictive_mean, color=color, lw=2.0)
        ax.scatter(dates, selected.observed, color=color, alpha=0.7, s=16, edgecolors="white", linewidths=0.35, zorder=4)
        ax.set_title(label, loc="left", color=color, fontweight="bold", fontsize=11, pad=8)
        ax.text(1, 1.035, f"WAPE = {metrics.wape_percent:.1f}%   |   95% PI coverage = {metrics.coverage_95_percent:.1f}%",
                transform=ax.transAxes, ha="right", va="bottom", color="#465661", fontsize=9.5)
        ax.set_ylabel("Cases / day")
        ax.set_ylim(0, max(selected.upper_95.max(), selected.observed.max()) * 1.08)
        ax.set_xlim(pd.Timestamp(START), pd.Timestamp(end_date))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax.grid(axis="y", color="#E4E8EB", lw=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#C6CDD3")
    fig.suptitle("Cases  |  England", fontsize=18, y=0.981)
    fig.text(0.5, 0.947, f"{START:%d %b %Y} – {end_date:%d %b %Y}  |  Historical fit comparison",
             ha="center", fontsize=10, color="#5D6871")
    handles = [Line2D([], [], color="#34495E", lw=2,
                      label="Model simulated under governmental actions\nMean of the predictive distribution"),
               Patch(facecolor="#8195A5", alpha=0.3, edgecolor="none", label="95% predictive interval"),
               Line2D([], [], color="#5D6871", marker="o", linestyle="none", markersize=4, label="Real data")]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.925), frameon=False, fontsize=9.0)
    axes[-1].set_xlabel("Date")
    fig.subplots_adjust(left=0.105, right=0.98, bottom=0.08, top=0.838, hspace=0.43)
    fig.savefig(OUTPUT / f"{filename_stem}.png", dpi=180)
    fig.savefig(OUTPUT / f"{filename_stem}.pdf")
    plt.close(fig)


def main():
    global OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BASE / "outputs")
    parser.add_argument("--verify-reproducibility", action="store_true")
    args = parser.parse_args()
    OUTPUT = args.output.resolve()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    settings = dict(SETTINGS)
    observed = pd.read_csv(DATA_PATH, parse_dates=["date"]).set_index("date")
    dates = [day.date() for day in observed.index]
    expected = [START + timedelta(days=i) for i in range((END - START).days + 1)]
    if dates != expected:
        raise ValueError("Real data must contain each day from 12 Oct through 31 Jan exactly once")
    values = observed[list(OBS_COLUMNS)]
    if values.isna().any().any() or (values < 0).any().any() or not np.equal(values, np.floor(values)).all().all():
        raise ValueError("Observed surveillance counts must be complete nonnegative integers")
    y = values.to_numpy(dtype=np.int64)
    initial_model = Model.from_dict(INITIAL_MODEL_PARAMETERS)
    updated_model = Model.from_dict(UPDATED_MODEL_PARAMETERS)
    contacts, contact_rows = contact_calendar(initial_model, dates)
    predictions, filtering, checks = rolling_filter(
        initial_model, INITIAL_PRIOR, updated_model, UPDATED_PRIOR,
        settings, dates, y, contacts, settings["seed"])
    if args.verify_reproducibility:
        repeat, repeat_filtering, _ = rolling_filter(
            initial_model, INITIAL_PRIOR, updated_model, UPDATED_PRIOR,
            settings, dates, y, contacts, settings["seed"])
        if not predictions.equals(repeat) or not filtering.equals(repeat_filtering):
            raise RuntimeError("Fixed-seed reproduction failed")
        print("Second run matches exactly", flush=True)
    scores = score_predictions(predictions)
    predictions.to_csv(OUTPUT / "daily_predictive_distributions.csv", index=False)
    filtering.to_csv(OUTPUT / "filter_diagnostics.csv", index=False)
    pd.DataFrame(contact_rows).to_csv(OUTPUT / "government_contact_calendar.csv", index=False)
    scores.to_csv(OUTPUT / "fit_metrics_by_age.csv", index=False)
    cases = predictions.loc[predictions.stream.str.startswith("cases_")].copy()
    cases.insert(2, "age_group", cases.stream.str.removeprefix("cases_"))
    cases.to_csv(OUTPUT / "plotted_cases.csv", index=False)
    draw_plot(predictions, scores)
    short_end = date(2021, 1, 15)
    short_predictions = predictions.loc[predictions.date.le(str(short_end))].copy()
    short_scores = score_predictions(short_predictions, end_date=short_end)
    short_scores.to_csv(OUTPUT / "fit_metrics_by_age_to_15jan.csv", index=False)
    draw_plot(short_predictions, short_scores, end_date=short_end,
              filename_stem="cases_model_fit_by_age_to_15jan")
    write_json(OUTPUT / "run_metadata.json", {
        "date_range": [str(START), str(END)], "settings": settings,
        "initial_fit_dates": ["2020-10-12", "2020-11-01"],
        "updated_fit_dates": ["2020-12-01", "2021-01-31"],
        "updated_parameters_from": str(NEW_PARAMETERS_FROM),
        "solid_line": "Mean of the predictive distribution",
        "parameters_estimated_during_this_run": False,
        "data_sha256": checksum(DATA_PATH.read_bytes()),
        "script_sha256": checksum(Path(__file__).read_bytes()),
        "fixed_seed_repetition_verified": args.verify_reproducibility,
        "forecast_timing_checks": checks})
    print(scores.loc[scores.evaluation_window.eq("all_requested_dates"),
                     ["age_group", "days", "wape_percent", "coverage_95_percent"]].to_string(index=False))
    print("Saved outputs to", OUTPUT, flush=True)


if __name__ == "__main__":
    main()

