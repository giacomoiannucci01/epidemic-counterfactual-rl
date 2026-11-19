"""Command-line entry point; all scientific model parameters are frozen."""
from __future__ import annotations

import argparse
from pathlib import Path
import json

def country_name(value):
    return {"uk":"england", "be":"belgium"}.get(value, value)


def parser():
    p=argparse.ArgumentParser(description="Particle-filtered epidemic counterfactuals: England or Belgium")
    p.add_argument('command',choices=['check','historical-fit','run','plot-run','reproduce-saved'])
    p.add_argument('--country',choices=['england','uk','belgium','be'],required=True,
                   help='uk is an alias for England data; this is not a UK-wide dataset')
    p.add_argument('--output',type=Path)
    p.add_argument('--profile',choices=['smoke','parallel-smoke','research'],default='smoke')
    p.add_argument('--particles',type=int)
    p.add_argument('--planning-particles',type=int)
    p.add_argument('--episodes',type=int,help='Explicit override: use this many episodes at each of the seven decisions')
    p.add_argument('--episodes-by-decision',type=int,nargs=7,
                   help='Explicit seven-count override; research defaults to 50000 47000 44000 41000 38000 34000 30000')
    p.add_argument('--iterations',type=int)
    p.add_argument('--mc-candidates',type=int)
    p.add_argument('--mc-rollouts',type=int)
    p.add_argument('--workers',type=int,help='Training rollout processes; research default 8')
    p.add_argument('--fit-workers',type=int,help='Independent fitted-Q ensemble processes; research default 3')
    p.add_argument('--mc-workers',type=int,help='MC candidate-action processes; research default 8')
    p.add_argument('--eval-workers',type=int,help='Independent evaluation processes; research default 8')
    p.add_argument('--eval-episodes',type=int)
    p.add_argument('--eval-mc-rollouts',type=int)
    p.add_argument('--start-method',choices=['auto','fork','spawn','forkserver'],default='auto')
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
        from .engine import Engine, study_calendar
        e=Engine(country)
        manifest={'country':country,'observed_rows':len(e.observed),'observed_channels':e.obs_columns,
                  'calendar':study_calendar(),
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
    config=(RunConfig.research(country) if args.profile=='research'
            else RunConfig.smoke(country,parallel=args.profile=='parallel-smoke'))
    for arg,field in [('particles','deployment_particles'),('planning_particles','planning_particles'),
                      ('iterations','outer_iterations'),
                      ('mc_candidates','mc_candidates'),('mc_rollouts','mc_rollouts'),('seed','seed'),
                      ('planning_trace_limit','planning_trace_limit'),('workers','workers'),
                      ('fit_workers','fit_workers'),('mc_workers','mc_workers'),('eval_workers','eval_workers'),
                      ('eval_episodes','eval_episodes'),('eval_mc_rollouts','eval_mc_rollouts'),('start_method','start_method')]:
        value=getattr(args,arg)
        if value is not None:setattr(config,field,value)
    if args.episodes is not None and args.episodes_by_decision is not None:
        raise ValueError('Choose --episodes or --episodes-by-decision')
    if args.episodes is not None:config.episodes_by_decision=(args.episodes,)*7
    if args.episodes_by_decision is not None:config.episodes_by_decision=tuple(args.episodes_by_decision)
    config.oracle=not args.no_oracle
    Experiment(country,config,output).run()
    totals=run_figures(country,output)
    print(json.dumps(totals,indent=2));print('Run outputs:',output)


if __name__=='__main__':
    main()
