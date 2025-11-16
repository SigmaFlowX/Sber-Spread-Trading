import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from numba import njit
from dateutil.relativedelta import relativedelta
import optuna

def open_data():
    df_sber = pd.read_csv("data/SBER10min.csv", parse_dates=["timestamp"], index_col="timestamp")
    df_sberp = pd.read_csv("data/SBERP10min.csv", parse_dates=["timestamp"], index_col="timestamp")

    df_sber = df_sber[['close']].copy()
    df_sberp = df_sberp[['close']].copy()

    df_sber.rename(columns={'close': 'SBER'}, inplace=True)
    df_sberp.rename(columns={'close': 'SBERP'}, inplace=True)

    df = pd.concat([df_sber, df_sberp], axis=1).dropna()

    return df

def prepare_data_arrays(df, z_window, spread_window):
    df = df.copy()

    y = df['SBER']
    X = sm.add_constant(df['SBERP'])

    rols = RollingOLS(y, X, window=spread_window)
    rres = rols.fit()

    params = rres.params
    df['a'] = params['SBERP']
    df['b'] = params['const']
    df['spread'] = y - (df['a'] * df['SBERP'] + df['b'])

    df['spread_mean'] = df['spread'].rolling(z_window).mean()
    df['spread_std'] = df['spread'].rolling(z_window).std()
    df['z_score'] = (df['spread'] - df['spread_mean']) / df['spread_std']

    df = df.dropna()

    return df['SBER'].values, df['SBERP'].values, df['z_score'].values, df['a'].values



@njit()
def run_strategy(sber_price_arr, sberp_price_arr, z_score_arr, a_arr, z_threshold):

    balance = 1000000
    risk_percent = 10
    pos = 0
    # SBER = a * SBERP + b
    # d(SBER) = a * d(SBERP)
    # SPREAD = SBER - a * SBERP - b
    # z > 0 - short a * SBER, long SBERP
    # z < 0 - long a * SBER, short SBERP
    for i in range(len(sberp_price_arr)):
        sber_price = sber_price_arr[i]
        sberp_price = sberp_price_arr[i]
        a = a_arr[i]
        z_score = z_score_arr[i]

        if pos == 0:
            if z_score > z_threshold:
                pos = 1
                total_pos_size = balance * risk_percent / 100
                sber_pos_size = a/(a+1) * total_pos_size
                sberp_pos_size = total_pos_size/(a+1)

                sber_quantity = sber_pos_size // sber_price
                sberp_quantity = sberp_pos_size // sberp_price
                sber_entry_price = sber_price
                sberp_entry_price = sberp_price

            elif z_score < -z_threshold:
                pos = -1

                total_pos_size = balance * risk_percent / 100
                sber_pos_size = a / (a + 1) * total_pos_size
                sberp_pos_size = total_pos_size / (a + 1)

                sber_quantity = sber_pos_size // sber_price
                sberp_quantity = sberp_pos_size // sberp_price
                sber_entry_price = sber_price
                sberp_entry_price = sberp_price
        else:
            if pos == 1 and z_score <= 0:
                pos = 0

                long_pnl = (sberp_price - sberp_entry_price) * sberp_quantity
                short_pnl = (sber_entry_price - sber_price) * sber_quantity
                balance += long_pnl + short_pnl

            elif pos == -1 and z_score >= 0:
                pos = 0

                long_pnl = (sber_price - sber_entry_price) * sber_quantity
                short_pnl = (sberp_entry_price - sberp_price) * sberp_quantity
                balance += long_pnl + short_pnl

    return balance

def objective(trial, df):
    df = df.copy()
    z_threshold = trial.suggest_float('z_threshold', 0.5,5)
    z_window = trial.suggest_int('z_window', 5,50)
    spread_window = trial.suggest_int('spread_window', 10,3000, log=True)

    sber_price_arr, sberp_price_arr, z_score_arr, a_arr = prepare_data_arrays(df, z_window, spread_window)

    return run_strategy(sber_price_arr, sberp_price_arr, z_score_arr, a_arr, z_threshold)

def generate_walkforward_windows(df, train_months=6, test_months=3):
    windows = []
    start_date = df.index.min()
    end_date = df.index.max()

    current_start = start_date

    while True:
        train_start = current_start
        train_end = train_start + relativedelta(months=train_months)
        test_start = train_end
        test_end = test_start + relativedelta(months=test_months)

        if test_end > end_date:
            break

        windows.append((train_start, train_end, test_start, test_end))
        current_start = test_start

    return windows



# Numba walk-forward optimization
test_results = []
if __name__ == "__main__":
    df = open_data()
    windows = generate_walkforward_windows(df)

    for train_start, train_end, test_start, test_end in windows:
        train_df = df.loc[train_start:train_end].copy()
        test_df = df.loc[test_start:test_end].copy()

        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, train_df), n_trials=100, n_jobs=1)

        best_params = study.best_params
        z_threshold = best_params['z_threshold']
        spread_window = best_params['spread_window']
        z_window = best_params['z_window']

        sber_price_arr, sberp_price_arr, z_score_arr, a_arr = prepare_data_arrays(test_df, z_window, spread_window)
        test_balance = run_strategy(sber_price_arr, sberp_price_arr, z_score_arr, a_arr, z_threshold)
        test_results.append(test_balance)

    print(test_results)