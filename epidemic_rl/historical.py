"""Frozen-parameter rolling predictive distributions; never re-fit parameters."""
from datetime import date, timedelta
from pathlib import Path
import json

import numpy as np
import pandas as pd

from .engine import Engine, AGES, COMPARTMENTS, weighted_summary
from .plotting import historical_cases, state_uncertainty


def scores_from_predictions(frame):
    rows=[]
    for stream,d in frame.groupby('stream',sort=False):
        rows.append({'stream':stream,'days':len(d),
                     'wape_percent':float(100*np.abs(d.observed-d.predictive_mean).sum()/max(d.observed.sum(),1)),
                     'coverage_95_percent':float(100*((d.observed>=d.lower_95)&(d.observed<=d.upper_95)).mean())})
    return pd.DataFrame(rows)


def england(output, particles=5000, draws=20000, seed=20261114):
    # Retain the literal historical-fit parameters and Dec6 switch/replay,
    # adding the common/centered-age log-q process used by Belgium and RL.
    from legacy.england_fit import reproduce as old
    from . import england_historical as current
    from .parameters_england import TRANSMISSION_PROCESS
    from .transmission import TransmissionProcess
    import hashlib
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    observed=pd.read_csv(old.DATA_PATH,parse_dates=['date']).set_index('date')
    dates=[d.date() for d in observed.index]
    if dates != pd.date_range(old.START,old.END).date.tolist():
        raise ValueError('England historical data must contain each day from 12 Oct through 31 Jan exactly once')
    y=observed[list(old.OBS_COLUMNS)].to_numpy(np.int64)
    initial=old.Model.from_dict(old.INITIAL_MODEL_PARAMETERS)
    updated=old.Model.from_dict(old.UPDATED_MODEL_PARAMETERS)
    contacts,calendar=old.contact_calendar(initial,dates)
    settings={**old.SETTINGS,'particles':particles,'predictive_draws':draws,'seed':seed}
    rows=[]
    def record(day,cloud,model):
        values=cloud.x.reshape(cloud.x.shape[0],-1)
        w=np.exp(cloud.logw);w/=w.sum();med,sd=weighted_summary(values,w)
        row={'date':str(day)}
        for g,a in enumerate(AGES):
            for j,state in enumerate(COMPARTMENTS):
                row[f'{a}_{state}_median']=float(med[g*8+j]);row[f'{a}_{state}_std']=float(sd[g*8+j])
        qmed,qsd=weighted_summary(np.exp(cloud.logq),w)
        for g,a in enumerate(AGES):
            row[f'{a}_q_median']=float(qmed[g]);row[f'{a}_q_std']=float(qsd[g])
        rows.append(row)
    predictions,diagnostics,checks,replay_diagnostics=current.rolling_filter(initial,old.INITIAL_PRIOR,updated,old.UPDATED_PRIOR,
        settings,dates,y,contacts,seed,state_callback=record)
    scores=old.score_predictions(predictions)
    predictions.to_csv(output/'daily_predictive_distributions.csv',index=False)
    diagnostics.to_csv(output/'filter_diagnostics.csv',index=False)
    diagnostic_summary=current.diagnostic_summary(diagnostics)
    diagnostic_summary.to_csv(output/'filter_diagnostics_summary.csv',index=False)
    replay_diagnostics.to_csv(output/'december_replay_filter_diagnostics.csv',index=False)
    scores.to_csv(output/'fit_metrics_by_age.csv',index=False)
    states=pd.DataFrame(rows);states.to_csv(output/'posterior_states.csv',index=False)
    pd.DataFrame(calendar).to_csv(output/'government_contact_calendar.csv',index=False)
    old.OUTPUT=output
    old.draw_plot(predictions,scores)
    current.plot_diagnostics(diagnostics,output)
    state_uncertainty(states,'england',output)
    metadata={'country':'england','parameters_fitted_during_run':False,'particles':particles,
              'predictive_draws':draws,'seed':seed,'forecast_before_current_observation_assimilation':True,
              'initial_fit_dates':['2020-10-12','2020-11-01'],
              'updated_fit_dates':['2020-12-01','2021-01-31'],'updated_parameters_apply_from':'2020-12-06',
              'historical_fit_profile':'Frozen England historical parameters with common/centered-age stochastic log-q, distinct from V18 RL clinical profile',
              'transmission_process':TransmissionProcess(**TRANSMISSION_PROCESS).manifest(),
              'initial_model_parameters':old.INITIAL_MODEL_PARAMETERS,
              'updated_model_parameters':old.UPDATED_MODEL_PARAMETERS,
              'xi':1.0,'separate_transmission_multiplier':False,
              'resampling_method':'systematic','ess_threshold_fraction':settings['ess_threshold_fraction'],
              'filter_diagnostics_summary':diagnostic_summary.to_dict(orient='records'),
              'data_sha256':hashlib.sha256(old.DATA_PATH.read_bytes()).hexdigest(),
              'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                  (Path(current.__file__),Path(__file__),Path(old.__file__),Path(__file__).with_name('transmission.py'),
                   Path(__file__).with_name('parameters_england.py'),Path(__file__).with_name('engine.py'))},
              'oracle_available':False,'checks':checks}
    (output/'historical_fit_manifest.json').write_text(json.dumps(metadata,indent=2))
    return metadata


def belgium(output, particles=5000, draws=20000, seed=2026):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    engine=Engine('belgium');cloud=engine.initial_belief(particles,seed)
    transition_rng,predict_rng,resample_rng=[np.random.default_rng(s) for s in np.random.SeedSequence([seed,448]).spawn(3)]
    rows=[];states=[];diagnostics=[]
    for timestamp in pd.date_range('2020-10-12','2021-01-31'):
        dt=timestamp.date();action=engine.government_action(dt)
        engine.predict(cloud,action,dt,transition_rng)
        # No current/future surveillance value enters forecast construction.
        idx=predict_rng.choice(cloud.n,size=draws,replace=True,p=cloud.weights)
        predictive_cloud=cloud.subset(idx)
        samples=engine.draw_observations(predictive_cloud,dt,predict_rng)
        mean=samples.mean(axis=0);sd=samples.std(axis=0)
        quantile=np.quantile(samples,[.025,.5,.975],axis=0)
        forecast=[{'date':str(dt),'forecast_origin':str(dt-timedelta(days=1)),
                   'stream':column,'predictive_mean':float(mean[j]),'predictive_median':float(quantile[1,j]),
                   'predictive_sd':float(sd[j]),'lower_95':float(quantile[0,j]),'upper_95':float(quantile[2,j])}
                  for j,column in enumerate(engine.obs_columns)]
        y=engine.observed.loc[timestamp,engine.obs_columns].to_numpy(float)
        for j,row in enumerate(forecast):row['observed']=int(y[j])
        rows.extend(forecast)
        diagnostic=engine.assimilate(cloud,y,dt,resample_rng,'historical')
        diagnostics.append({'date':str(dt),**diagnostic})
        states.append({'date':str(dt),**engine.state_row(cloud)})
    predictions=pd.DataFrame(rows);scores=scores_from_predictions(predictions)
    predictions.to_csv(output/'daily_predictive_distributions.csv',index=False)
    scores.to_csv(output/'fit_metrics_by_age.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(output/'filter_diagnostics.csv',index=False)
    states=pd.DataFrame(states);states.to_csv(output/'posterior_states.csv',index=False)
    historical_cases(predictions,'belgium',engine.case_ages,scores,output)
    state_uncertainty(states,'belgium',output)
    metadata={'country':'belgium','parameters_fitted_during_run':False,'particles':particles,
              'predictive_draws':draws,'seed':seed,'forecast_before_current_observation_assimilation':True,
              'initial_fit_dates':['2020-09-01','2020-09-21'],
              'updated_fit_dates':['2020-11-03','2020-11-30'],'updated_parameters_apply_from':'2020-12-01',
              'transmission_process':engine.transmission_process.manifest(),
              'xi':1.0,'separate_transmission_multiplier':False,
              'oracle_available':False,'historical_q_not_forced_from_future_fit':True}
    (output/'historical_fit_manifest.json').write_text(json.dumps(metadata,indent=2))
    return metadata
