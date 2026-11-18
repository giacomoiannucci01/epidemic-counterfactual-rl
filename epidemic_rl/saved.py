"""Reproduce figures from supplied saved analyses, separately from new training."""
from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt

from .engine import ROOT, AGES
from .plotting import counterfactual, deaths_figure, save, style, COLORS


def reproduce(country, output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    reference=ROOT/'reference/saved_analysis'
    rl=pd.read_csv(reference/f'{country}.csv')
    observed=pd.read_csv(ROOT/'data'/country/'pf_observations.csv')
    label='RL counterfactual (observation mean)' if country=='belgium' else 'RL counterfactual'
    counterfactual(country,rl,observed,output,case_label=label)
    deaths=pd.read_csv(ROOT/'data'/country/('deaths_total.csv' if country=='england' else 'deaths.csv'))
    deaths_figure(country,rl,deaths,output)
    if country=='england':
        adjusted=rl.copy();adjusted['deaths']=rl.deaths_display_initial_aligned
        deaths_figure(country,adjusted,deaths,output,display_aligned=True,
                       name='deaths_real_vs_rl_initial_value_aligned')
    gov=pd.read_csv(ROOT/'data'/country/'government_actions.csv')
    dates=pd.to_datetime(rl.date);control=dates.le('2021-01-15')
    for display_copy in ([False,True] if country=='belgium' else [False]):
        fig,axes=plt.subplots(3,1,figsize=(10.5,9),sharex=True)
        shown=rl.copy()
        if display_copy:
            edited=pd.read_csv(reference/'belgium_displayed_actions.csv').set_index('date')
            shown=shown.set_index('date')
            for age in AGES:
                shown.loc[edited.index,'a_'+age]=edited['a_'+age]
            shown=shown.reset_index()
        for age,color,ax in zip(AGES,COLORS,axes):
            g=gov.loc[gov.date.le('2021-01-15')]
            ax.step(pd.to_datetime(g.date),g['a_'+age],where='post',color=color,lw=1.8,label='Government actions')
            ax.step(dates[control],shown.loc[control,'a_'+age],where='post',color=color,lw=1.8,ls='--',
                    label='RL actions (display edits)' if display_copy else 'Saved RL actions')
            ax.set_title(age,loc='left',color=color);ax.set_ylabel('Contact multiplier');style(ax);ax.set_ylim(0,1.05)
        fig.suptitle(country.title()+' — Actions'+(' | Display edits' if display_copy else ''),y=.985)
        handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=2,loc='upper center',bbox_to_anchor=(.5,.95),frameon=False)
        fig.subplots_adjust(top=.87,hspace=.3);axes[-1].set_xlabel('Date')
        save(fig,output,'actions_display_copy' if display_copy else 'actions_saved_policy')
    source=json.loads((reference/'provenance.json').read_text())
    metadata={'country':country,'uses_saved_trajectories':True,'new_training_performed':False,
              'new_filter_updates_performed':False,'source':source[country],
              'period_deaths_model':int(rl.deaths.sum()),
              'period_icu_patient_days':int(rl.icu_occupancy.sum()),
              'plot_end':'2021-01-31','totals_end':'2021-03-12'}
    (output/'saved_analysis_manifest.json').write_text(json.dumps(metadata,indent=2))
    rl.to_csv(output/'saved_daily_values.csv',index=False)
    return metadata
