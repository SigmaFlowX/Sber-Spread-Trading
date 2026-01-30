import strategy
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from numba import njit
from dateutil.relativedelta import relativedelta
import optuna
import matplotlib.pyplot as plt
from datetime import timedelta
import numpy as np

STARTING_BALANCE = 100000
FEE = 0.01/100
SINCE = "01-01-2024" #None to use all the data
TIMEFRAME = 10  #1 or 10 (min)
N_TRIALS = 30      #optuna study trials
N_TRAIN_MONTHS = 6
N_TEST_MONTHS = 3
PLOT_EQUITIES = False
RISK_PCT = 10

def prepare_df(df, z_window, spread_window):
    df = df.copy()

    y = df['SBER']
    x = sm.add_constant(df['SBERP'])

    rols = RollingOLS(y, x, window=spread_window)
    rres = rols.fit()

    params = rres.params
    df['a'] = params['SBERP']
    df['b'] = params['const']
    df['spread'] = y - (df['a'] * df['SBERP'] + df['b'])

    df['spread_mean'] = df['spread'].rolling(z_window).mean()
    df['spread_std'] = df['spread'].rolling(z_window).std()
    df['z_score'] = (df['spread'] - df['spread_mean']) / df['spread_std']

    df = df.dropna()

    return df

def faster_backtest(df, z_threshold,initial_balance=STARTING_BALANCE, test=False):
    df['signal'] = np.where(
        df['z_score'] > z_threshold, 1,
        np.where(df['z_score'] < -z_threshold, -1, 0)
    )

    df['position'] = (
        df['signal']
        .replace(0, np.nan)
        .where(np.sign(df['z_score']) == df['signal'])
        .ffill()
        .fillna(0)
    )

    df['position_exec'] = df['position'].shift(1).fillna(0)

    df['pair_ret'] = (
            - df['position_exec'] * df['a'] * df['SBER'].pct_change()
            + df['position_exec'] * df['SBERP'].pct_change()
    )

    df['balance_prev'] = initial_balance if 'balance' not in df else df['balance'].shift(1).fillna(initial_balance)
    df['exposure'] = RISK_PCT * df['balance_prev']

    df['gross_pnl'] = df['exposure'] * df['pair_ret']

    df['pos_change'] = df['position_exec'].diff().abs()
    df['turnover'] = df['pos_change'] * df['exposure'] * (1 + df['a'])
    df['commission'] = FEE * df['turnover']

    df['pnl'] = df['gross_pnl'] - df['commission']
    df['balance'] = initial_balance + df['pnl'].cumsum()

    final_balance = df.iloc[-1]['balance']

    if test:
        equity_curve = pd.Series(df['balance'], index=df.index)
        return equity_curve, (final_balance - initial_balance)/initial_balance * 100, final_balance

    return (final_balance - initial_balance)/initial_balance * 100

def objective(trial, df):
    df = df.copy()
    z_threshold = trial.suggest_float('z_threshold', 0.5, 5)
    z_window = trial.suggest_int('z_window', 5, 400, log=True)
    spread_window = trial.suggest_int('spread_window', 10, 3000, log=True)

    data = prepare_df(df, z_window, spread_window)

    return faster_backtest(data, z_threshold)



if __name__ == "__main__":
    row_df = strategy.open_data(timeframe=TIMEFRAME, since=SINCE)
    df = prepare_df(row_df, 10, 10)

    windows = strategy.generate_walkforward_windows(df, train_months=N_TRAIN_MONTHS, test_months=N_TEST_MONTHS)

    balance = STARTING_BALANCE
    for train_start, train_end, test_start, test_end in windows:
        print( f"\nPeriod: {train_start.date()} — {train_end.date()} (train), {test_start.date()} — {test_end.date()} (test)")
        train_df = df.loc[train_start:train_end].copy()
        test_df = df.loc[test_start:test_end].copy()

        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, train_df), n_trials=N_TRIALS, n_jobs=8)

        best_params = study.best_params
        z_threshold = best_params['z_threshold']
        spread_window = best_params['spread_window']
        z_window = best_params['z_window']

        test_data = prepare_df(test_df, z_window, spread_window)
        equity_series, total_return, final_balance = faster_backtest(test_data, z_threshold, initial_balance=balance, test=True)
        print(f"Total test return is {total_return:.2f}%")
        print("------------------------------------------")
        balance = final_balance

