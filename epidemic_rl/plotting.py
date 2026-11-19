"""Scientific figures from numerical outputs, without cosmetic data changes."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

COLORS = ("#0072B2", "#D55E00", "#009E73")


def style(ax):
    ax.grid(axis="y", color="#E4E8EB", linewidth=.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_ylim(bottom=0)


def save(fig, output, name):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(output/f"{name}.{extension}", dpi=180, facecolor="white")
    plt.close(fig)


def historical_cases(predictions, country, case_ages, scores, output):
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 9), sharex=True)
    for age, color, ax in zip(case_ages, COLORS, axes):
        d = predictions.loc[predictions.stream.eq("cases_"+age)]
        dt = pd.to_datetime(d.date)
        ax.fill_between(dt, d.lower_95, d.upper_95, color=color, alpha=.18, label="95% predictive interval")
        ax.plot(dt, d.predictive_mean, color=color, lw=1.9, label="Mean of predictive distribution")
        ax.scatter(dt, d.observed, s=10, color="#44515B", alpha=.7, label="Real data", zorder=4)
        s = scores.loc[scores.stream.eq("cases_"+age)].iloc[0]
        label = age.replace("_plus", "+").replace("_", "–")
        ax.set_title(label, loc="left", color=color)
        ax.text(.985, .92, f"WAPE {s.wape_percent:.1f}% | Coverage {s.coverage_95_percent:.1f}%",
                transform=ax.transAxes, ha="right", fontsize=9)
        ax.set_ylabel("Cases / day"); style(ax)
    fig.suptitle(f"{country.title()} — Historical particle-filter fit", fontsize=16, y=.99)
    fig.text(.5, .95, "12 Oct 2020 – 31 Jan 2021 | Rolling one-day-ahead comparison", ha="center")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5,.93), ncol=3, frameon=False)
    axes[-1].set_xlabel("Date")
    fig.subplots_adjust(top=.85, bottom=.08, left=.1, right=.98, hspace=.3)
    save(fig, output, "cases_model_fit_by_age")


def state_uncertainty(states, country, output, run_label=None):
    states = states.copy(); dates = pd.to_datetime(states.date)
    compartments = ("S", "E1", "E2", "I1", "I2", "H", "U", "R")
    for age in ("0_44", "45_64", "65_plus"):
        fig, axes = plt.subplots(4, 2, figsize=(12, 10), sharex=True)
        for state, ax in zip(compartments, axes.ravel()):
            median = states[f"{age}_{state}_median"]
            sd = states[f"{age}_{state}_std"]
            ax.plot(dates, median, color=COLORS[0], lw=1.2)
            ax.fill_between(dates, np.maximum(0,median-sd), median+sd, color=COLORS[0], alpha=.2)
            ax.set_title(state); style(ax)
        fig.suptitle(f"{country.title()} | {age} | PF state median and ±1 standard deviation", fontsize=13)
        if run_label:
            fig.text(.5,.015,run_label,ha='center',fontsize=9)
        fig.tight_layout(rect=(0,0,1,.96))
        save(fig, output, f"state_uncertainty_{age}")


def counterfactual(country, rl, observed, output, *, plot_end="2021-01-31", case_label="RL counterfactual", run_label=None):
    """Observed reporting ages are kept in their native bands."""
    rl = rl.copy(); observed = observed.copy()
    rl["date"] = pd.to_datetime(rl.date); observed["date"] = pd.to_datetime(observed.date)
    r = rl.loc[rl.date.le(pd.Timestamp(plot_end))]
    o = observed.loc[observed.date.between(r.date.min(),r.date.max())]
    ages = ("0_44","45_64","65_plus") if country=="england" else ("0_39","40_59","60_plus")
    fig, axes = plt.subplots(3,1,figsize=(10.5,9),sharex=True)
    for age, color, ax in zip(ages,COLORS,axes):
        ax.plot(r.date,r["cases_"+age],color=color,lw=1.8,label=case_label)
        ax.scatter(o.date,o["cases_"+age],s=12,color=color,alpha=.65,label="Real data")
        ax.set_title(age,loc="left",color=color);ax.set_ylabel("Cases / day");style(ax)
    fig.suptitle(country.title()+" — Real cases versus RL counterfactual",fontsize=15,y=.985)
    if run_label:
        fig.text(.5,.915,run_label,ha='center',fontsize=9)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc="upper center",ncol=2,bbox_to_anchor=(.5,.95),frameon=False)
    axes[-1].set_xlabel("Date");fig.subplots_adjust(top=.87,hspace=.3)
    save(fig,output,"cases_real_vs_rl_by_age")
    stock = "mv_beds_all_ages" if country=="england" else "icu_occupied_total"
    fig,ax=plt.subplots(figsize=(10.5,4.8))
    ax.plot(r.date,r[stock],color=COLORS[0],lw=1.9,label="RL counterfactual")
    ax.scatter(o.date,o[stock],color="#52616D",s=14,label="Real data")
    ax.set_title(country.title()+(' — Mechanical ventilation beds' if country=='england' else ' — ICU occupancy'))
    ax.set_ylabel("Patients / occupied beds");ax.set_xlabel("Date");style(ax);ax.legend(frameon=False)
    if run_label:
        fig.suptitle(run_label,fontsize=9)
    fig.tight_layout();save(fig,output,"ventilation_real_vs_rl" if country=='england' else "icu_real_vs_rl")


def deaths_figure(country, rl, observed, output, *, display_aligned=False, name='deaths_real_vs_rl', run_label=None):
    r=rl.copy();o=observed.copy();r['date']=pd.to_datetime(r.date);o['date']=pd.to_datetime(o.date)
    r=r.loc[r.date.le('2021-01-31')];o=o.loc[o.date.between(r.date.min(),r.date.max())]
    fig,ax=plt.subplots(figsize=(10.5,4.8))
    ax.plot(r.date,r.deaths,color=COLORS[0],lw=1.9,label='RL counterfactual'+(' (initial value aligned)' if display_aligned else ''))
    ax.scatter(o.date,o.deaths_total,color='#52616D',s=14,label='Real data')
    ax.set_title(country.title()+' — Daily deaths');ax.set_ylabel('Deaths / day');ax.set_xlabel('Date')
    if run_label:
        fig.suptitle(run_label,fontsize=9)
    style(ax);ax.legend(frameon=False);fig.tight_layout();save(fig,output,name)


def policy_figures(country, rl, government, output, summary, run_label=None):
    """Actions and budget evolution through the end of control, 15 January."""
    r=rl.loc[rl.date.le('2021-01-15')].copy()
    g=government.loc[government.date.le('2021-01-15')].copy()
    ages=('0_44','45_64','65_plus')
    fig,axes=plt.subplots(3,1,figsize=(10.5,9),sharex=True)
    for age,color,ax in zip(ages,COLORS,axes):
        for frame,ls,label in [(g,'-','Government actions'),(r,'--','RL actions')]:
            ax.step(pd.to_datetime(frame.date),frame['a_'+age],where='post',color=color,ls=ls,lw=1.8,label=label)
        ax.set_title(age.replace('_plus','+').replace('_','–'),loc='left',color=color)
        ax.set_ylabel('Contact multiplier');style(ax);ax.set_ylim(0,1.05)
    fig.suptitle(country.title()+' — Actions',y=.985)
    if run_label:fig.text(.5,.915,run_label,ha='center',fontsize=9)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=2,loc='upper center',bbox_to_anchor=(.5,.95),frameon=False)
    axes[-1].set_xlabel('Date');fig.subplots_adjust(top=.87,hspace=.3)
    save(fig,output,'actions_government_vs_rl')
    budget=float(summary['social_budget'])
    ref=summary['government_reference'];alpha=summary['config']['alpha']
    dr=max(ref['deaths_in_period']+ref['pending_expected_deaths_after_period'],1.)
    ur=max(ref['icu_patient_days'],1.)
    fig,axes=plt.subplots(2,1,figsize=(10.5,7),sharex=True)
    rows=[]
    for frame,ls,label in [(g,'-','Government'),(r,'--','RL')]:
        remaining=budget-frame.social_cost.cumsum()
        health=alpha*frame.deaths.cumsum()/dr+(1-alpha)*frame.icu_occupancy.cumsum()/ur
        axes[0].plot(pd.to_datetime(frame.date),remaining,color=COLORS[0],ls=ls,lw=1.8,label=label)
        axes[1].plot(pd.to_datetime(frame.date),health,color=COLORS[1],ls=ls,lw=1.8,label=label)
        rows.extend({'date':day,'arm':label,'remaining_social_budget':float(b),
                     'cumulative_normalized_health_loss':float(h)} for day,b,h in zip(frame.date,remaining,health))
    for ax in axes:style(ax);ax.legend(frameon=False)
    # Lagrangian spending may exceed the target; preserve a negative balance.
    lower=min(row['remaining_social_budget'] for row in rows)
    if lower<0:axes[0].set_ylim(bottom=lower*1.05)
    axes[0].set_ylabel('Remaining social budget');axes[1].set_ylabel('Cumulative health loss')
    axes[-1].set_xlabel('Date');fig.suptitle(country.title()+' — Budget and health loss')
    if run_label:fig.text(.5,.02,run_label,ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.95));save(fig,output,'budget_and_health_loss')
    pd.DataFrame(rows).to_csv(Path(output)/'budget_and_health_loss.csv',index=False)


def run_figures(country, output):
    """Generate every iteration's figures from an existing completed run."""
    from .paths import ROOT
    output=Path(output)
    summary=json.loads((output/'run_summary.json').read_text())
    if summary['country']!=country:raise ValueError('Run country does not match --country')
    observed=pd.read_csv(ROOT/'data'/country/'pf_observations.csv')
    deaths=pd.read_csv(ROOT/'data'/country/('deaths_total.csv' if country=='england' else 'deaths.csv'))
    government=pd.read_csv(output/'government_daily.csv')
    dual=pd.read_csv(output/'dual_trace.csv')
    # Compatibility with completed runs made before this metadata field existed.
    summary.setdefault('social_budget',float(dual.budget.iloc[0]))
    for row in dual.itertuples(index=False):
        folder=output/f'iter_{row.iteration:02d}';rl=pd.read_csv(folder/'rl_daily.csv')
        label=f'Iteration {row.iteration} | lambda {row.lambda_used:.3f}'
        if summary['config']['profile']!='research':label+=' | Smoke test: reduced particles and training'
        counterfactual(country,rl,observed,folder,run_label=label)
        state_uncertainty(pd.read_csv(folder/'belief_states.csv'),country,folder,run_label=label)
        deaths_figure(country,rl,deaths,folder,run_label=label)
        policy_figures(country,rl,government,folder,summary,run_label=label)
        totals={'start_date':rl.date.min(),'end_date':rl.date.max(),
                'rl_deaths_in_period':int(rl.deaths.sum()),
                'rl_icu_patient_days':int(rl.icu_occupancy.sum()),
                'observed_deaths_in_period':int(deaths.loc[deaths.date.between(rl.date.min(),rl.date.max()),'deaths_total'].sum())}
        (folder/'period_totals.json').write_text(json.dumps(totals,indent=2))
    return totals
