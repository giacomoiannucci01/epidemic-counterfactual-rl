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
    calendar=summary['calendar']
    assert calendar=={'control_start':'2020-10-12','control_end':'2021-01-17',
                      'control_days':98,'decision_lengths':[14]*7,'tail_start':'2021-01-18',
                      'tail_days':54,'study_end':'2021-03-12','study_days':152}
    assert summary['per_decision_fits']==7*config['outer_iterations']
    assert summary['state_feature_count']==83
    assert summary['oracle_changes_posterior_uncertainty'] is False
    process=summary['transmission_process']
    assert process['daily_log_drift']==[0.,0.,0.]
    assert process['common_daily_sd']==.1 and process['raw_age_daily_sd']==.08
    expected=np.full((3,3),.01-.0064/3)
    np.fill_diagonal(expected,.01+2*.0064/3)
    np.testing.assert_allclose(process['daily_log_innovation_covariance'],expected,rtol=1e-14)
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
        daily=pd.read_csv(iteration/'rl_daily.csv')
        assert daily.date.tolist()==pd.date_range('2020-10-12','2021-03-12').strftime('%Y-%m-%d').tolist()
        decisions=pd.read_csv(iteration/'decisions_and_policy_state.csv')
        assert len(decisions)==7
        assert decisions.duration_days.tolist()==[14]*7
        assert decisions.date.tolist()==pd.date_range('2020-10-12',periods=7,freq='14D').strftime('%Y-%m-%d').tolist()
        assert decisions.end_date.iloc[-1]=='2021-01-17'
        assert len(daily.loc[daily.date.le('2021-01-17')])==98
        tail=daily.loc[daily.date.ge('2021-01-18')]
        assert len(tail)==54 and (tail.social_cost==0).all()
        for age in ('0_44','45_64','65_plus'):
            final=daily.loc[daily.date.between('2021-01-04','2021-01-17'),'a_'+age]
            np.testing.assert_array_equal(final.to_numpy(),np.repeat(decisions['a_'+age].iloc[-1],14))
        budget=pd.read_csv(iteration/'budget_and_health_loss.csv')
        assert budget.date.max()=='2021-01-17'
        assert len(budget.loc[budget.arm.eq('RL')])==98
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
