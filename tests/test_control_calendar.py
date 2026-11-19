"""Seven full fortnight decisions followed by a 54-day common health tail."""
from datetime import date, timedelta
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from epidemic_rl.engine import (Engine, START, CONTROL_DAYS, CONTROL_END,
    DECISION_LENGTHS, TAIL_START, TAIL_DAYS, END, STUDY_DAYS, study_calendar)
from epidemic_rl.planning import RunConfig, Simulation


class ControlCalendar(unittest.TestCase):
    def test_seven_full_decisions_and_54_tail_days_partition_study(self):
        self.assertEqual(DECISION_LENGTHS,(14,)*7)
        self.assertEqual(CONTROL_DAYS,98)
        self.assertEqual(CONTROL_END,date(2021,1,17))
        self.assertEqual(TAIL_START,date(2021,1,18))
        self.assertEqual(TAIL_DAYS,54)
        self.assertEqual(STUDY_DAYS,152)
        control=[]
        for decision,duration in enumerate(DECISION_LENGTHS):
            start=START+timedelta(days=sum(DECISION_LENGTHS[:decision]))
            control.extend(start+timedelta(days=j) for j in range(duration))
        tail=[timestamp.date() for timestamp in pd.date_range(TAIL_START,END)]
        self.assertEqual(control[-14:],[timestamp.date() for timestamp in pd.date_range('2021-01-04','2021-01-17')])
        self.assertEqual(control+tail,[timestamp.date() for timestamp in pd.date_range(START,END)])
        self.assertEqual(len(set(control+tail)),152)
        self.assertEqual(study_calendar()['decision_lengths'],[14]*7)

    def test_time_feature_and_government_budget_use_all_98_control_days(self):
        for country in ('england','belgium'):
            with self.subTest(country=country):
                engine=Engine(country);belief=engine.initial_belief(8,2026)
                index=engine.feature_names().index('remaining_control_fraction')
                self.assertEqual(engine.features(belief,[],98,0)[index],1.)
                np.testing.assert_allclose(engine.features(belief,[],14,0)[index],14/98,rtol=1e-7)
                self.assertEqual(engine.features(belief,[],0,0)[index],0.)
                total=sum(engine.contact_loss(engine.government_action(t.date()))*engine.cost_scale
                          for t in pd.date_range(START,CONTROL_END))
                self.assertAlmostEqual(total,engine.budget,places=10)

    def test_last_decision_runs_through_jan17_before_54_government_tail_days(self):
        for country in ('england','belgium'):
            with self.subTest(country=country):
                engine=Engine(country);config=RunConfig.smoke(country)
                belief=engine.initial_belief(8,2026)
                truth=engine.sample_truth(belief,np.random.default_rng(10))
                rng_t=np.random.default_rng(11);rng_p=np.random.default_rng(12)
                sim=Simulation(engine,config)
                action=np.array([.4,.7,.2])
                with patch.object(engine,'step_with_filter',wraps=engine.step_with_filter) as spy:
                    truth,belief,_,_,_,control,_=sim.block(truth,belief,action,date(2021,1,4),
                        DECISION_LENGTHS[-1],rng_t,rng_p,collect=True)
                    _,_,_,_,tail,_=sim.tail(truth,belief,rng_t,rng_p,collect=True)
                self.assertEqual(len(control),14);self.assertEqual(len(tail),54)
                self.assertEqual(control[-1]['date'],'2021-01-17')
                self.assertEqual(tail[0]['date'],'2021-01-18')
                self.assertEqual(tail[-1]['date'],'2021-03-12')
                self.assertTrue(all(row['social_cost']>0 for row in control))
                self.assertTrue(all(row['social_cost']==0 for row in tail))
                self.assertEqual(engine.counters['planning']['pf_updates'],68)
                self.assertEqual([call.args[3] for call in spy.call_args_list],
                                 [t.date() for t in pd.date_range('2021-01-04','2021-03-12')])
                for call in spy.call_args_list[:14]:
                    np.testing.assert_array_equal(call.args[2],action)
                for call in spy.call_args_list[14:]:
                    np.testing.assert_array_equal(call.args[2],engine.government_action(call.args[3]))


if __name__=='__main__':unittest.main()
