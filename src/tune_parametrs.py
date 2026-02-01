# This code is used to find current optimal parameters for the strategy
from src import strategy
import optuna


df = strategy.open_data(timeframe=1, since="2025-06-01")
study = optuna.create_study(direction="maximize")
study.optimize(lambda trial: strategy.objective(trial, df), n_trials=200, n_jobs=8)
best_params = study.best_params
print(best_params)