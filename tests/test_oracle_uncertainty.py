"""Point-only oracle correction must never collapse PF uncertainty."""
from datetime import timedelta
import pickle
import unittest
from unittest.mock import patch

import numpy as np

from epidemic_rl.engine import Engine, START
from epidemic_rl.planning import RunConfig, _context_digest


class OracleUncertainty(unittest.TestCase):
    def setup_cloud(self, country):
        engine=Engine(country)
        belief=engine.initial_belief(24,2026)
        truth=engine.sample_truth(belief,np.random.default_rng(3))
        belief.logq[:]=np.log(1e-7)+np.linspace(-.5,.5,belief.n)[:,None]
        belief.logw[:]=np.log(np.arange(1,belief.n+1)/sum(range(1,belief.n+1)))
        return engine,belief,truth

    def assert_particles_equal(self, a, b):
        for (name,x),(_,y) in zip(a.particle_arrays(),b.particle_arrays()):
            np.testing.assert_array_equal(x,y,err_msg=name)

    def test_all_policy_centers_change_and_all_pf_standard_deviations_remain(self):
        for country in ('england','belgium'):
            with self.subTest(country=country):
                e,b,t=self.setup_cloud(country);before=b.copy()
                med,sd=e.state_statistics(b)
                b,triggers=e.oracle_gate(b,t,phase='planning')
                self.assertTrue(triggers);self.assert_particles_equal(b,before)
                point,policy_sd=e.policy_statistics(b)
                np.testing.assert_array_equal(point,e.state_matrix(t)[0])
                np.testing.assert_array_equal(policy_sd,sd)
                np.testing.assert_array_equal(e.state_statistics(b)[0],med)
                row=e.state_row(b)
                self.assertTrue(row['oracle_point_corrected'])
                for g,age in enumerate(('0_44','45_64','65_plus')):
                    for j,state in enumerate(e.state_names()):
                        self.assertEqual(row[f'{age}_{state}_std'],sd[g*12+j])
                        self.assertEqual(row[f'{age}_{state}_median'],med[g*12+j])
                        self.assertEqual(row[f'{age}_{state}_policy_estimate'],point[g*12+j])

    def test_copies_worker_payloads_and_planning_root_keep_correction_separate(self):
        for country in ('england','belgium'):
            e,b,t=self.setup_cloud(country);b,_=e.oracle_gate(b,t)
            variants=[b.copy(),b.subset(np.arange(12)),pickle.loads(pickle.dumps(b))]
            for child in variants:
                np.testing.assert_array_equal(e.policy_statistics(child)[0],e.state_matrix(t)[0])
                np.testing.assert_array_equal(e.policy_statistics(child)[1],e.state_statistics(child)[1])
                self.assertFalse(np.shares_memory(child.oracle_state.x,b.oracle_state.x))
            sampled=e.sample_truth(b,np.random.default_rng(8))
            self.assertEqual(sampled.n,1);self.assertIsNone(sampled.oracle_state)
            self.assert_particles_equal(sampled,t)
            self.assertFalse(np.shares_memory(sampled.x,b.oracle_state.x))

    def test_point_correction_expires_before_new_propagation_and_when_disabled(self):
        for country in ('england','belgium'):
            e,b,t=self.setup_cloud(country);b,_=e.oracle_gate(b,t)
            e.predict(b,e.government_action(START),START,np.random.default_rng(9))
            self.assertIsNone(b.oracle_state)
            for left,right in zip(e.policy_statistics(b),e.state_statistics(b)):
                np.testing.assert_array_equal(left,right)
            b,_=e.oracle_gate(b,t)
            self.assertIsNotNone(b.oracle_state)
            before=b.copy();b,triggers=e.oracle_gate(b,t,enabled=False)
            self.assertFalse(triggers);self.assertIsNone(b.oracle_state)
            self.assert_particles_equal(b,before)

    def test_daily_filter_has_same_posterior_with_or_without_point_oracle(self):
        for country in ('england','belgium'):
            e,b,t=self.setup_cloud(country);action=e.government_action(START)
            baseline=e.step_with_filter(t.copy(),b.copy(),action,START,
                np.random.default_rng(21),np.random.default_rng(22),phase='planning',oracle=False)
            corrected=e.step_with_filter(t.copy(),b.copy(),action,START,
                np.random.default_rng(21),np.random.default_rng(22),phase='planning',oracle=True)
            self.assertTrue(corrected[-1]);self.assert_particles_equal(baseline[1],corrected[1])
            np.testing.assert_array_equal(e.state_statistics(baseline[1])[1],e.policy_statistics(corrected[1])[1])
            np.testing.assert_array_equal(e.policy_statistics(corrected[1])[0],e.state_matrix(corrected[0])[0])

    def test_failed_weight_update_preserves_predictive_uncertainty(self):
        for country in ('england','belgium'):
            e,b,t=self.setup_cloud(country);expected=b.copy();action=e.government_action(START)
            e.predict(expected,action,START,np.random.default_rng(42))
            with patch.object(e,'assimilate',side_effect=RuntimeError('Particle weights collapsed')):
                truth,belief,y,_,diagnostic,triggers=e.step_with_filter(t,b,action,START,
                    np.random.default_rng(41),np.random.default_rng(42),phase='planning')
            e.condition_observation_states(expected,y,START)
            self.assert_particles_equal(belief,expected)
            np.testing.assert_array_equal(e.policy_statistics(belief)[1],e.state_statistics(expected)[1])
            self.assertTrue(np.any(e.policy_statistics(belief)[1]>0))
            np.testing.assert_array_equal(e.policy_statistics(belief)[0],e.state_matrix(truth)[0])
            self.assertEqual(triggers,['numeric_filter_collapse'])
            self.assertTrue(diagnostic['observation_update_failed'])
            self.assertEqual(e.counters['planning']['pf_updates'],1)

    def test_checkpoint_identity_includes_current_point_correction(self):
        e,b,t=self.setup_cloud('england')
        context=dict(engine=e,config=RunConfig(),belief=b,decision=0,spent=0.,history=[],
                     lam=.5,references=(10000.,100000.),seed=3,code_digest='oracle-test')
        original=_context_digest(context)
        e.oracle_gate(b,t)
        corrected=_context_digest(context)
        self.assertNotEqual(original,corrected)
        self.assertEqual(corrected,_context_digest(pickle.loads(pickle.dumps(context))))


if __name__=='__main__':unittest.main()
