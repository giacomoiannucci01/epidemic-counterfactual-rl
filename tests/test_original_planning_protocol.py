"""Protocol restoration and worker-invariant stochastic/FQI calculations."""
import ast
import copy
from dataclasses import fields
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from epidemic_rl.engine import Engine, ROOT
from epidemic_rl.fitted_q import Dataset, QNetwork, fit_ddqn
from epidemic_rl.planning import RunConfig, generate_dataset, select_mc, rank_mc_results


class OriginalPlanningProtocol(unittest.TestCase):
    def test_country_original_mc_tie_breaking(self):
        for country,expected in [('england',2),('belgium',7)]:
            rows=[{'row':{'mean_mc_return':-10.,'action_id':i}} for i in [7,2]]
            rank_mc_results(rows,country)
            self.assertEqual(rows[0]['row']['action_id'],expected)

    def test_research_schedule_and_country_fitting_settings(self):
        for country in ('england','belgium'):
            cfg=RunConfig.research(country);cfg.validate()
            self.assertEqual(cfg.episodes_by_decision,(50000,47000,44000,41000,38000,34000,30000))
            self.assertEqual((cfg.workers,cfg.fit_workers,cfg.mc_workers,cfg.eval_workers),(8,3,8,8))
            self.assertEqual((cfg.fqi_iterations,cfg.epochs_per_iteration,cfg.ensemble_size),(36,3,3))
            self.assertEqual((cfg.mc_candidates,cfg.mc_rollouts,cfg.eval_episodes,cfg.eval_mc_rollouts),(64,192,16,32))
            self.assertEqual((cfg.deployment_particles,cfg.planning_particles),(5000,1024))
            expected=(512,1e-5,2,1) if country=='england' else (2048,1e-4,4,12)
            self.assertEqual((cfg.batch_size,cfg.weight_decay,cfg.target_sync_every,cfg.government_replicates),expected)

    def test_fitted_q_matches_supplied_original_member_exactly(self):
        rng=np.random.default_rng(14)
        ds=Dataset(rng.normal(size=(24,83)).astype('float32'),rng.integers(64,size=24),
                   rng.normal(size=24).astype('float32'),rng.normal(size=(24,83)).astype('float32'),
                   np.array([0,1]*12,dtype='float32'),np.ones((24,64),bool),np.arange(24),np.zeros(24))
        for country in ('england','belgium'):
            cfg=RunConfig.smoke(country);cfg.batch_size=8
            actual,_,_=fit_ddqn(ds,cfg.fit_config(),43,country,workers=1)
            mean=ds.states.mean(0).astype('float32');sd=np.maximum(ds.states.std(0),1e-6).astype('float32')
            context=dict(X=(ds.states-mean)/sd,XN=(ds.next_states-mean)/sd,A=ds.actions,
                         D=ds.done,R=ds.rewards,config=cfg.fit_config(),seed=43,members=1,
                         n_actions=64,checkpoint_dir=None,digest=None)
            if country=='england':
                filename='england_fit_parallel.py.txt';function='_member_fit'
                context['next_mask']=ds.next_mask;key='_CONTEXT'
            else:
                filename='belgium_fitted_q.py.txt';function='_fit_one_member'
                context['masks']=ds.next_mask;key='_FIT_CTX'
            source=ROOT/'provenance/original_kernels'/filename
            tree=ast.parse(source.read_text(encoding='utf-8-sig'))
            node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==function)
            scope=dict(np=np,torch=torch,nn=nn,DataLoader=DataLoader,TensorDataset=TensorDataset,
                       copy=copy,os=os,time=time,QNetwork=QNetwork,planner=SimpleNamespace(QNetwork=QNetwork))
            scope[key]=context
            exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),scope)
            _,expected,_=scope[function](0)
            for name,value in actual.nets[0].state_dict().items():
                np.testing.assert_array_equal(value.numpy(),np.asarray(expected[name]),err_msg=f'{country}: {name}')

    def test_training_fitting_and_mc_are_worker_invariant_with_pf(self):
        for country in ('england','belgium'):
            with self.subTest(country=country), tempfile.TemporaryDirectory() as temporary:
                cfg=RunConfig.smoke(country);cfg.planning_particles=8;cfg.ensemble_size=2
                cfg.episode_chunk_size=2;cfg.batch_size=4
                engine=Engine(country);belief=engine.initial_belief(12,2026)
                context=dict(engine=engine,config=cfg,belief=belief,decision=5,
                             history=[np.array([.7,.4,1.])]*5,spent=8.,lam=.8,
                             references=(20000.,200000.),seed=143,code_digest='protocol-test')
                a,stats_a,trace_a=generate_dataset(context,4,workers=1,
                                                 checkpoint=Path(temporary)/'chunks',trace_limit=10)
                b,stats_b,trace_b=generate_dataset(context,4,workers=2,trace_limit=10)
                for field in fields(a):
                    np.testing.assert_array_equal(getattr(a,field.name),getattr(b,field.name))
                self.assertEqual(len(a),8)
                self.assertEqual(sum(s['counters']['planning']['pf_updates'] for s in stats_a),4*82)
                self.assertEqual([s['counters'] for s in stats_a],[s['counters'] for s in stats_b])
                self.assertEqual(trace_a,trace_b)
                self.assertIn('65_plus_U_std',trace_a[0])
                changed=dict(context,lam=.9)
                with self.assertRaisesRegex(RuntimeError,'Checkpoint inputs differ'):
                    generate_dataset(changed,4,checkpoint=Path(temporary)/'chunks')
                one,_,_=fit_ddqn(a,cfg.fit_config(),194,country,workers=1)
                many,_,audit=fit_ddqn(a,cfg.fit_config(),194,country,workers=2)
                self.assertEqual(audit['workers_requested'],2)
                for n1,n2 in zip(one.nets,many.nets):
                    for k,v in n1.state_dict().items():
                        np.testing.assert_array_equal(v.numpy(),n2.state_dict()[k].numpy())
                context.update(policy=one.payload(),rollouts=1)
                ai,serial=select_mc(context,workers=1,trace_limit=10)
                bi,parallel=select_mc(context,workers=2,trace_limit=10)
                self.assertEqual(ai,bi)
                self.assertEqual([r['row'] for r in serial],[r['row'] for r in parallel])
                self.assertEqual([r['counters'] for r in serial],[r['counters'] for r in parallel])
                self.assertGreater(sum(r['counters']['planning']['pf_updates'] for r in parallel),0)


if __name__=='__main__':unittest.main()
