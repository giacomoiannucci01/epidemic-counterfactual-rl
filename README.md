# Epidemic counterfactual RL: England and Belgium

This repository runs the existing eight-compartment stochastic epidemic model with fixed, previously estimated parameters. Choose England or Belgium. Training, hypothetical Monte Carlo planning and policy deployment all run a particle filter; the policy receives posterior **medians and standard deviations**, previous actions, remaining time and remaining budget.

The repository also reproduces the previously requested England historical particle-filter fit and provides the equivalent rolling one-day-ahead fit for Belgium. No command estimates epidemic, observation or mortality parameters. Optimisation during `run` trains the RL neural network only.

## Run it

Use Python 3.12. The validated environment used Python 3.12.5 and the versions pinned in `requirements.txt`.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install --no-deps .

python -m epidemic_rl check --country england
python -m epidemic_rl check --country belgium
```

`--country uk` is an alias for `england`: the actual surveillance geography is England, not the whole United Kingdom.

The wheel includes the required data and saved references, so the installed `epidemic-rl` command also works outside the checkout. Outputs default to the current working directory. On Linux, installing the CPU-only PyTorch wheel first avoids unnecessary CUDA dependencies: `python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu`.

Rebuild the historical predictive-fit figures, with real observations, predictive means and 95% predictive intervals:

```bash
python -m epidemic_rl historical-fit --country england --output outputs/england/historical
python -m epidemic_rl historical-fit --country belgium --output outputs/belgium/historical
```

The default historical filter uses 5,000 particles and 20,000 predictive observation draws. For a quick software check, use `--particles 128 --predictive-draws 1000`. Figures contain WAPE and empirical predictive-interval coverage; no R-squared is calculated or shown.

Run a complete small training/planning/deployment check:

```bash
python -m epidemic_rl run --country england --profile smoke --output outputs/england/smoke
python -m epidemic_rl run --country belgium --profile smoke --output outputs/belgium/smoke
```

Smoke runs deliberately use reduced particle and training counts and label their figures accordingly. They exercise the complete algorithm; they are not production scientific results.

Start a research-sized run:

```bash
python -m epidemic_rl run --country england --profile research --output outputs/england/research
python -m epidemic_rl run --country belgium --profile research --output outputs/belgium/research
```

The research profile uses 5,000 deployment particles, 1,024 planning particles, three DDQN ensemble members, 2,000 training episodes per outer iteration, 20 outer iterations, all 64 candidate actions and 192 PF Monte Carlo rollouts per candidate. Running a PF in every planning rollout makes this substantially more expensive than the original full-state fast planner. Particle counts, episodes, iterations and MC counts can be set explicitly with the CLI options. The full research profile has not been run to convergence as part of software packaging.

This is a new PF-planning variant. Its policy representation and training procedure differ from the previous fast full-state fitted-Q implementation. New training is not advertised as reproducing the old learned policies bit for bit.

Replay stores health and social costs separately and evaluates every training batch at the current Lagrange multiplier. Experiences collected before a price update therefore use the new objective consistently.

## State seen by RL

The named 83-component policy state contains:

- The weighted posterior median and weighted population standard deviation for each age group's `S, E1, E2, I1, I2, H, U, R, D, q`, plus the original combined `E` and `I` variables.
- The three most recent action vectors, each with its three age-specific multipliers. The three components of the immediately preceding action are retained.
- Remaining control-time fraction and remaining social-budget fraction. The budget fraction can become negative; this is a Lagrangian price target, not a hard action mask.

Count features are population-normalized; transmission features are scaled by the initial fitted q. `feature_schema.json` records the exact input order. `decisions_and_policy_state.csv` records the values actually passed to action selection. `belief_states.csv` records daily state medians and standard deviations in count units.

The PF uses normalized likelihood weights, an ESS threshold of half the particle count, and systematic resampling with one uniform offset. Standard deviations are calculated around the weighted mean, not around the median.

## PF during planning and truth recovery

Each hypothetical rollout samples its own hidden truth from the current belief. It generates synthetic surveillance observations under the candidate actions, advances a separate particle cloud, assimilates those synthetic observations and selects subsequent actions from the resulting median/SD state. It never takes features directly from a future deployment truth or assimilates future real surveillance into a counterfactual arm.

The requested truth fallback is a **simulation oracle**. It compares the current simulated truth with the current PF median. Source-based thresholds are E ratio 1.30, I ratio 1.25, H/U ratio 1.50, q ratio 1.20 and susceptible population-fraction error 0.0005, with small-count floors. If a threshold is exceeded, the complete cloud and conditioning buffers are reset to the current truth. That current state is then known to the policy, and its posterior SD is zero until subsequent process/observation uncertainty develops. Triggers, pre-reset ESS, reset flags and phase counters are recorded.

Use `--no-oracle` for the corresponding ablation. An oracle run is privileged simulation research, not a deployable real-world controller or an ordinary Bayesian posterior claim. Real-data historical filtering has no oracle.

`planning_belief_states.csv` contains medians/SDs from hypothetical planning. The trace is capped by default, while counters cover every PF update; set `--planning-trace-limit -1` to retain the full trace. A successful smoke run must perform both planning PF updates and neural-network updates.

## Frozen parameters and dates

All numerical values are literals in Python source, with `#` comments recording their fitting windows:

| Component | Source | Fitting window / application |
|---|---|---|
| England RL initial model | `epidemic_rl/parameters_england.py` | 12 Oct–1 Nov 2020 initial calibration |
| England RL late clinical/transmission model | same file | Existing world-0 fit for 15 Dec 2020–31 Jan 2021; applied on those dates, including planning futures |
| England mortality | same file | Previously fitted common multiplier 0.7494361261548264, using government totals over 12 Oct–31 Jan; all nine resulting probabilities are literals |
| Exact previous England historical-fit profile | `legacy/england_fit/reproduce.py` | Initial fit 12 Oct–1 Nov; updated fit 1 Dec–31 Jan, applied from 6 Dec, with the original Dec1–5 replay |
| Belgium initial model | `epidemic_rl/parameters_belgium.py` | Fit 1–21 Sep 2020, applied from 22 Sep |
| Belgium December model | same file | Fit 3–30 Nov 2020; applied from 1 Dec, including hypothetical planning futures |
| Belgium mortality | same file | Original UK-derived prior used by the saved Belgian experiment; no Belgium-specific mortality fit is asserted |

The exact England historical-fit profile and the V18 RL late-regime profile are separate existing analyses. They retain their respective fitted values and switch dates; they are not silently combined. Their calibrations are retrospective. Fixed fitted parameters are not claimed to be known prospectively before their fitting windows ended.

The model uses simultaneous, exclusive clinical exits. Deaths tag subsets of exits into R and do not remove a person from both R and the population a second time. The original transition kernel is included for audit and tested against the implementation using identical transition random numbers.

## Observations and Excel workbooks

- `data/england/real_observations.xlsx`: cases and admissions in 0–44, 45–64, 65+; all-age mechanical-ventilation occupancy; observed deaths. MV is not renamed ICU.
- `data/belgium/real_observations.xlsx`: cases in the actual reported 0–39, 40–59, 60+ bands; all-age admissions and ICU occupancy; age-specific deaths.

The CSV versions drive the code. Only required real channels are included. Belgium's bootstrap additionally needs 1 Sep–11 Oct observations; study data extend through 12 Mar where supplied. Missing age-specific ICU/admissions are not invented. The Belgian mapping from latent 0–44/45–64/65+ to reported case bands is the original fixed population-based mapping and is documented in its parameter file.

Government contact calendars and the previously reconstructed q scenarios are separate model inputs. q scenarios are not labelled as observed surveillance. They define the retrospective deployment world; the planning PF does not receive their future values. England's q scenario ends on 31 Jan and subsequently follows the original random-walk law. The saved Belgian scenario covers the full health tail.

UKHSA's direct total-death archive and its older age-specific COVerAGE snapshot are kept separate; small snapshot differences are not hidden by rescaling ages. See `DATA_SOURCES.md` and `provenance/parameter_sources.json`.

## Outputs and periods

The control period is 12 Oct 2020–15 Jan 2021: six 14-day decisions and a final 12-day decision. The common government-contact health tail runs 16 Jan–12 Mar. Figures show real versus counterfactual daily observations through 31 Jan. Period totals sum the 152 days through 12 Mar. Expected pending deaths after 12 Mar are separately recorded for the optimisation reference and are not added to observed-period death counts.

Each run writes government and RL daily trajectories, selected actions, policy states, particle-state median/SD tables, oracle logs, the dual trace, network checkpoints, case/death/ICU-or-MV comparisons and state-uncertainty figures. Historical fitting writes one-day-ahead predictive distributions, WAPE/coverage diagnostics and state median/SD figures.

Every iteration gets its own figures, including government/RL actions and remaining social budget through 15 Jan. Government lines are solid and RL lines dashed in the policy/budget comparisons, with the same colour for the same quantity. The health panel reports the cumulative normalized death/ICU loss used by the objective. Regenerate figures without training again using `python -m epidemic_rl plot-run --country england --output outputs/england/research` (or `belgium` and its run folder).

## Reproduce the previously saved analyses

```bash
python -m epidemic_rl reproduce-saved --country england --output outputs/england/saved
python -m epidemic_rl reproduce-saved --country belgium --output outputs/belgium/saved
```

These commands read reduced, saved source trajectories and do not claim to re-train them. They are distinct from `run`.

The previous Belgian presentation labelled source iteration 6 as iteration 10 and changed the displayed 65+ action in decision 6 from 0.2 to 0.7. The saved clinical outputs were not regenerated for that display edit. The repository therefore exports the actual saved policy figure and a clearly labelled display copy separately. England's optional initial-value-aligned death figure is likewise an illustrative display copy, not a new simulated trajectory. Numerical model totals remain separate from aligned-display totals.

## Tests and citation

```bash
python -m unittest discover -s tests -v
```

Tests check systematic resampling, weighted SDs, all policy-state fields, date switches, forecast timing, current-truth recovery and equality with the original transition kernel. `provenance/validation_results.json` records local numerical/reproduction checks. GitHub Actions also exercises complete smoke runs for both countries.

Historical-fit examples and their numerical diagnostics are in [`examples/`](examples/): [England](examples/england_historical_fit_reference.png) and [Belgium](examples/belgium_historical_fit.png).

Author: **Giacomo Iannucci**. `CITATION.cff` contains the software citation metadata for [the public repository](https://github.com/giacomoiannucci01/epidemic-counterfactual-rl). Cite a tagged release/commit for reproducibility. A DOI is not asserted; an archival DOI can be added after depositing a release.
