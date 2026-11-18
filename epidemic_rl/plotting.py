"""Scientific figures from numerical outputs, without cosmetic data changes."""
from pathlib import Path

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


def state_uncertainty(states, country, output):
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
