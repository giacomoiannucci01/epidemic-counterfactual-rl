"""Command-line entry point; all scientific model parameters are frozen."""
from __future__ import annotations

import argparse
from pathlib import Path
import json

import pandas as pd


def country_name(value):
    return {"uk":"england", "be":"belgium"}.get(value, value)


def parser():
    p=argparse.ArgumentParser(description="Particle-filtered epidemic counterfactuals: England or Belgium")
    p.add_argument('command',choices=['check','historical-fit','run','plot-run','reproduce-saved'])
    p.add_argument('--country',choices=['england','uk','belgium','be'],required=True,
                   help='uk is an alias for England data; this is not a UK-wide dataset')
    p.add_argument('--output',type=Path)
    p.add_argument('--profile',choices=['smoke','research'],default='smoke')
    p.add_argument('--particles',type=int)
    p.add_argument('--planning-particles',type=int)
    p.add_argument('--episodes',type=int)
    p.add_argument('--iterations',type=int)
    p.add_argument('--mc-candidates',type=int)
    p.add_argument('--mc-rollouts',type=int)
    p.add_argument('--predictive-draws',type=int,default=20000)
    p.add_argument('--seed',type=int)
    p.add_argument('--no-oracle',action='store_true',help='Disable simulator-truth recovery (ablation)')
    p.add_argument('--planning-trace-limit',type=int)
    return p


def main():
    args=parser().parse_args();country=country_name(args.country)
    output=(args.output or Path.cwd()/'outputs'/country/('run' if args.command=='plot-run' else args.command)).resolve()
    if args.command=='plot-run':
        from .plotting import run_figures
        print(json.dumps(run_figures(country,output),indent=2));return
    if args.command=='check':
        from .engine import Engine
        e=Engine(country)
        manifest={'country':country,'observed_rows':len(e.observed),'observed_channels':e.obs_columns,
                  'actions_calendar_rows':len(e.gov),'policy_state_features':len(e.feature_names()),
                  'statistics_in_policy_state':['median','standard deviation'],
                  'action_history_lags':3,'model_parameter_estimation_at_runtime':False}
        print(json.dumps(manifest,indent=2));return
    if args.command=='historical-fit':
        from . import historical
        kwargs={'particles':args.particles or 5000,'draws':args.predictive_draws}
        if args.seed is not None:kwargs['seed']=args.seed
        getattr(historical,country)(output,**kwargs)
        print('Historical PF figures and diagnostics:',output);return
    if args.command=='reproduce-saved':
        from .saved import reproduce
        reproduce(country,output);print('Saved-analysis figures:',output);return
    from .planning import RunConfig,Experiment
    from .plotting import run_figures
    config=RunConfig.research() if args.profile=='research' else RunConfig()
    for arg,field in [('particles','deployment_particles'),('planning_particles','planning_particles'),
                      ('episodes','train_episodes'),('iterations','outer_iterations'),
                      ('mc_candidates','mc_candidates'),('mc_rollouts','mc_rollouts'),('seed','seed'),
                      ('planning_trace_limit','planning_trace_limit')]:
        value=getattr(args,arg)
        if value is not None:setattr(config,field,value)
    config.oracle=not args.no_oracle
    Experiment(country,config,output).run()
    totals=run_figures(country,output)
    print(json.dumps(totals,indent=2));print('Run outputs:',output)


if __name__=='__main__':
    main()
