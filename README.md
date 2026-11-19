# Epidemic counterfactual RL: England and Belgium

This repository runs the existing eight-compartment stochastic epidemic model with fixed, previously estimated parameters. Choose England or Belgium. A particle filter estimates the epidemic state throughout RL training and decision-making. The policy uses the posterior median and standard deviation of each state, together with previous actions, remaining time and remaining budget.

Install and test
Use Python 3.12.
git clone https://github.com/giacomoiannucci01/epidemic-counterfactual-rl.git
cd epidemic-counterfactual-rl
python -m venv .venv
Activate the environment:
- Linux/macOS: source .venv/bin/activate
- Windows PowerShell: .\.venv\Scripts\Activate.ps1
Install dependencies and run the tests:
python -m pip install -r requirements.txt
python -m pip install --no-deps .
python -m unittest discover -s tests -v
Run the analysis
Choose England or Belgium:
# England
python -m epidemic_rl run --country england --profile research --output outputs/england/research

# Belgium
python -m epidemic_rl run --country belgium --profile research --output outputs/belgium/research
For a quick check of the complete workflow:
python -m epidemic_rl run --country england --profile parallel-smoke --output outputs/england/smoke
For Belgium, replace england with belgium in both arguments.
Optional: historical model fit
Generate the historical case-fit figure and numerical diagnostics:
python -m epidemic_rl historical-fit --country england --output outputs/england/historical
Use belgium instead of england for Belgium. Outputs include predictive means, 95% predictive intervals, WAPE and empirical interval coverage.
Results
The commands above save results in outputs/england/research/ or outputs/belgium/research/.
- run_summary.json — configuration and execution diagnostics.
- dual_trace.csv — lambda updates and social spending.
- planning_belief_states.csv — state medians and standard deviations during planning.
- iter_01/ … iter_20/ — trajectories, actions, state uncertainty, period totals and PNG/PDF figures.
- iter_XX/decision_YY/ — fitted networks, training checkpoints and Monte Carlo diagnostics.
Change --output to choose another destination.
Data and model inputs
Required CSV and Excel observations are included in data/england/ and data/belgium/. Fixed epidemic parameters and their fitting dates are recorded in epidemic_rl/parameters_england.py and epidemic_rl/parameters_belgium.py.

The policy receives state medians and standard deviations, the three previous action vectors, remaining time and remaining budget.
Citation
Author: Giacomo Iannucci. Use CITATION.cff and record the commit used for your experiment.
