import ast
from datetime import date
from pathlib import Path
import unittest
from types import SimpleNamespace

import numpy as np

from epidemic_rl.engine import Engine, Cloud, START, ROOT, CONTROL_DAYS, weighted_summary, systematic_resample


class ScientificContracts(unittest.TestCase):
    def test_social_reward_uses_current_fixed_dual_price(self):
        from epidemic_rl.planning import health_reward, dual_update
        a=health_reward(2,3,5,1,(100,200),30,.5)
        b=health_reward(2,3,5,2,(100,200),30,.5)
        self.assertAlmostEqual(b-a,-98*5/30)
        self.assertEqual(dual_update(.5,30,30,.4,1),(.5,.4))

    def test_systematic_resampling_mass_and_reproducibility(self):
        np.testing.assert_array_equal(systematic_resample([0,0,1,0],np.random.default_rng(1)),[2,2,2,2])
        a=systematic_resample([.1,.2,.3,.4],np.random.default_rng(4))
        b=systematic_resample([.1,.2,.3,.4],np.random.default_rng(4))
        np.testing.assert_array_equal(a,b)
        rng=np.random.default_rng(4)
        frequencies=np.bincount(np.concatenate([systematic_resample([.1,.2,.3,.4],rng) for _ in range(3000)]),minlength=4)/12000
        np.testing.assert_allclose(frequencies,[.1,.2,.3,.4],rtol=0,atol=.008)

    def test_posterior_sd_is_around_weighted_mean_not_median(self):
        median,sd=weighted_summary(np.array([[0.,0.],[10.,20.]]),np.array([.75,.25]))
        np.testing.assert_array_equal(median,[0,0])
        np.testing.assert_allclose(sd,[np.sqrt(18.75),np.sqrt(75.)])

    def test_each_country_features_and_calendar_switch(self):
        for country in ['england','belgium']:
            with self.subTest(country=country):
                e=Engine(country);b=e.initial_belief(24,2026)
                names=e.feature_names();f=e.features(b,[np.array([.2,.4,.7])],CONTROL_DAYS,0)
                self.assertEqual(len(names),83);self.assertEqual(len(f),83)
                for age in ['0_44','45_64','65_plus']:
                    for state in ['S','E1','E2','I1','I2','H','U','R','D','q']:
                        self.assertIn(f'{age}_{state}_median',names);self.assertIn(f'{age}_{state}_std',names)
                self.assertEqual(f[names.index('previous_action_lag1_65_plus')],np.float32(.7))
                if country=='belgium':
                    self.assertFalse(np.array_equal(e.model(date(2020,11,30)).p_hospital,e.model(date(2020,12,1)).p_hospital))
                else:
                    self.assertFalse(np.array_equal(e.model(date(2020,12,14)).p_icu,e.model(date(2020,12,15)).p_icu))

    def test_forecast_does_not_read_current_surveillance(self):
        for country in ['england','belgium']:
            e=Engine(country);a=e.initial_belief(16,2026);b=a.copy()
            action=e.government_action(START)
            e.predict(a,action,START,np.random.default_rng(71))
            ya=e.draw_observations(a,START,np.random.default_rng(72))
            e.observed.loc[:,:]=99999999
            e.predict(b,action,START,np.random.default_rng(71))
            yb=e.draw_observations(b,START,np.random.default_rng(72))
            np.testing.assert_array_equal(ya,yb)
            np.testing.assert_array_equal(a.x,b.x)

    def test_oracle_corrects_point_and_preserves_particle_uncertainty(self):
        for country in ['england','belgium']:
            e=Engine(country);b=e.initial_belief(16,2026)
            t=e.sample_truth(b,np.random.default_rng(3))
            b.logq[:]=np.log(1e-7)+np.linspace(-.5,.5,b.n)[:,None]
            before=b.copy();median,sd=e.state_statistics(b)
            features=e.features(b,[],CONTROL_DAYS,0)
            corrected,triggers=e.oracle_gate(b,t,phase='planning')
            self.assertTrue(triggers)
            for (name,a),(_,old) in zip(corrected.particle_arrays(),before.particle_arrays()):
                np.testing.assert_array_equal(a,old,err_msg=name)
            np.testing.assert_array_equal(e.state_statistics(corrected)[0],median)
            np.testing.assert_array_equal(e.state_statistics(corrected)[1],sd)
            np.testing.assert_array_equal(e.policy_statistics(corrected)[0],e.state_matrix(t)[0])
            np.testing.assert_array_equal(e.policy_statistics(corrected)[1],sd)
            self.assertTrue(np.any(sd>0))
            after=e.features(corrected,[],CONTROL_DAYS,0)
            np.testing.assert_array_equal(after[1:72:2],features[1:72:2])
            np.testing.assert_array_equal(after[72:],features[72:])
            self.assertEqual(e.counters['planning']['oracle_resets'],1)

    def test_model_transition_matches_original_kernel_exactly(self):
        # Compare all compartments/flows against the supplied original model
        # using identical transition RNG state (including preceding q draws).
        source=ROOT/'provenance/original_kernels/england_core.py.txt'
        tree=ast.parse(source.read_text(encoding='utf-8-sig'))
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_advance_controlled')
        scope={'np':np,**dict(zip(['S','E1','E2','I1','I2','H','U','R'],range(8)))}
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),scope)
        e=Engine('england');original=e.initial_belief(16,2026);test=original.copy()
        r1=np.random.default_rng(91);r2=np.random.default_rng(91)
        q=original.logq.copy()
        e.transmission_process.advance(q,r1)
        r1.normal(0,e.rho_sd,size=original.n)
        q=np.exp(q)
        m=e.model(START);action=np.array([.7,.4,.2]);C=e.contacts(action)
        # The archived reference accepts an extra factor; use its neutral value
        # solely for this comparison. The runtime model has no such parameter.
        reference=SimpleNamespace(**m.__dict__,transmission_scale=1.)
        expected,_,_=scope['_advance_controlled'](original.x,q,reference,r1,1.,C)
        e.predict(test,action,START,r2)
        np.testing.assert_array_equal(expected,test.x)


if __name__=='__main__':
    unittest.main()
