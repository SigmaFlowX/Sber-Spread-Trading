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
SINCE = "01-01-2023" #None to use all the data
TIMEFRAME = 1   #1 or 10 (min)
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

def faster_backtest(df, z_threshold):
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

    df['balance_prev'] = STARTING_BALANCE if 'balance' not in df else df['balance'].shift(1).fillna(STARTING_BALANCE)
    df['exposure'] = RISK_PCT * df['balance_prev']

    df['gross_pnl'] = df['exposure'] * df['pair_ret']

    df['pos_change'] = df['position_exec'].diff().abs()
    df['turnover'] = df['pos_change'] * df['exposure'] * (1 + df['a'])
    df['commission'] = FEE * df['turnover']

    df['pnl'] = df['gross_pnl'] - df['commission']
    df['balance'] = STARTING_BALANCE + df['pnl'].cumsum()



if __name__ == "__main__":
    row_df = strategy.open_data(timeframe=TIMEFRAME, since=SINCE)
    df = prepare_df(row_df, 10, 10)

    faster_backtest(df, 1)
