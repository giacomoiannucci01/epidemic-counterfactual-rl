"""The epidemic force uses the paper's q_g,t and no separate time factor."""
from datetime import date
import unittest

import numpy as np

from epidemic_rl.engine import Engine, I1, I2, S
from legacy.england_fit import reproduce as historical


class InfectionForce(unittest.TestCase):
    def test_force_matches_paper_sum_for_each_age_in_both_calibration_regimes(self):
        action=np.array([.7,.4,.2])
        dates=[date(2020,10,12),date(2020,11,30),date(2020,12,1),date(2020,12,22),
               date(2021,1,12),date(2021,2,1)]
        for country in ('england','belgium'):
            engine=Engine(country);cloud=engine.initial_belief(6,2026)
            cloud.logq[:]=np.log(np.array([[.03,.05,.08],[.04,.02,.09]]*3))
            for dt in dates:
                with self.subTest(country=country,date=dt):
                    model=engine.model(dt);contacts=engine.contacts(action,dt)
                    expected=np.array([[np.exp(cloud.logq[i,g])*sum(
                        contacts[g,h]*(cloud.x[i,h,I1]+model.xi*cloud.x[i,h,I2])/engine.population[h]
                        for h in range(3)) for g in range(3)] for i in range(cloud.n)])
                    np.testing.assert_allclose(engine.infection_force(cloud,action,dt),expected,
                                               rtol=1e-14,atol=0)
                    self.assertFalse(hasattr(model,'transmission_scale'))
                    self.assertNotIn('transmission_scale',model.to_dict())

    def test_historical_fit_transition_already_uses_q_without_extra_factor(self):
        for values in (historical.INITIAL_MODEL_PARAMETERS,historical.UPDATED_MODEL_PARAMETERS):
            model=historical.Model.from_dict(values)
            x=np.array(historical.INITIAL_PRIOR['mean_x'])[None,:,:]
            force=model.q*((x[:,:,I1]+model.xi*x[:,:,I2])/model.population @ model.contact_matrix.T)
            _,infections,_=historical.advance(x,model,rng=None)
            np.testing.assert_array_equal(infections,x[:,:,S]*(-np.expm1(-force)))


if __name__=='__main__':unittest.main()
