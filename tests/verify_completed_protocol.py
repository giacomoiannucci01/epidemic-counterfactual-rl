"""Check completed CLI runs, including real worker PIDs and dual evaluation."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


def verify(folder):
    folder=Path(folder)
    summary=json.loads((folder/'run_summary.json').read_text())
    config=summary['config']
    assert summary['per_decision_fits']==7*config['outer_iterations']
    assert summary['state_feature_count']==83
    assert summary['oracle_changes_posterior_uncertainty'] is False
    assert summary['dual_update_source']=='independent closed-loop policy evaluations'
    assert summary['pf_counters']['planning']['pf_updates']>0
    assert summary['pf_counters']['evaluation']['pf_updates']==config['eval_episodes']*152*config['outer_iterations']
    assert summary['planning_trace_rows_seen']==summary['pf_counters']['planning']['pf_updates']
    dual=pd.read_csv(folder/'dual_trace.csv')
    for row in dual.itertuples(index=False):
        iteration=folder/f'iter_{row.iteration:02d}'
        evals=json.loads((iteration/'independent_evaluations.json').read_text())
        assert len(evals)==config['eval_episodes']
        expected=np.mean([r['social_cost'] for r in evals])
        np.testing.assert_allclose(row.expected_policy_spend,expected,rtol=0,atol=1e-12)
        np.testing.assert_allclose(row.lambda_next,np.clip(row.lambda_used+row.eta*(expected/row.budget-1),0,4),rtol=0,atol=1e-12)
        assert all(r['mc_workers_inside_evaluation']==1 for r in evals)
        assert len(pd.read_csv(iteration/'rl_daily.csv'))==152
        decisions=pd.read_csv(iteration/'decisions_and_policy_state.csv')
        assert len(decisions)==7
        assert 'oracle_point_corrected' in decisions
        states=pd.read_csv(iteration/'belief_states.csv')
        assert '65_plus_U_policy_estimate' in states
        assert 'oracle_point_corrected' in states
        for decision in range(1,8):
            audit=json.loads((iteration/f'decision_{decision:02d}/timings.json').read_text())
            assert audit['training_episodes']==config['episodes_by_decision'][decision-1]
            assert audit['training_transitions']==audit['training_episodes']*(8-decision)
            for phase in ('training','fit','mc'):
                if config['profile']=='parallel-smoke':
                    assert len(set(audit[phase+'_pids']))>1,(phase,decision,audit)
        if config['profile']=='parallel-smoke':
            assert len({r['pid'] for r in evals})>1
    print(folder.name+': per-decision training, PF, independent dual evaluation and workers verified')
    return summary


if __name__=='__main__':
    for name in sys.argv[1:]:
        verify(name)
