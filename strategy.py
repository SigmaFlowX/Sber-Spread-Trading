import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from numba import njit
from dateutil.relativedelta import relativedelta
import optuna
import matplotlib.pyplot as plt
from datetime import timedelta
import numpy as np


#optuna.logging.set_verbosity(optuna.logging.CRITICAL)

def open_data(timeframe, since=None):
    df_sber = pd.read_csv(f"data/SBER{timeframe}min.csv", parse_dates=["timestamp"], index_col="timestamp")
    df_sberp = pd.read_csv(f"data/SBERP{timeframe}min.csv", parse_dates=["timestamp"], index_col="timestamp")

    df_sber = df_sber[['close']].copy()
    df_sberp = df_sberp[['close']].copy()

    df_sber.rename(columns={'close': 'SBER'}, inplace=True)
    df_sberp.rename(columns={'close': 'SBERP'}, inplace=True)

    df = pd.concat([df_sber, df_sberp], axis=1).dropna()

    if since is not None:
        df = df[df.index >= pd.to_datetime(since)]

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
def run_strategy_fast(sber_price_arr, sberp_price_arr, z_score_arr, a_arr, z_threshold):

    fee = 0.008/100
    initial_balance = 1000000
    balance = initial_balance
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
                total_fee = fee * (sber_quantity * sber_price + sberp_quantity * sberp_price) * 2
                balance += long_pnl + short_pnl - total_fee

            elif pos == -1 and z_score >= 0:
                pos = 0

                long_pnl = (sber_price - sber_entry_price) * sber_quantity
                short_pnl = (sberp_entry_price - sberp_price) * sberp_quantity
                total_fee = fee * (sber_quantity * sber_price + sberp_quantity * sberp_price) * 2
                balance += long_pnl + short_pnl - total_fee

    return (balance - initial_balance)/initial_balance * 100

def test_strategy_slow(sber_price_arr, sberp_price_arr, z_score_arr, a_arr, z_threshold, timestamps,plot=False):
    fee = 0.008 / 100
    initial_balance = 1000000
    balance = initial_balance
    risk_percent = 10
    pos = 0
    pnls = []
    equity_curve = []
    holding_times = []
    paid_fees = 0

    for i in range(len(sberp_price_arr)):
        sber_price = sber_price_arr[i]
        sberp_price = sberp_price_arr[i]
        a = a_arr[i]
        z_score = z_score_arr[i]
        time = timestamps[i]

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

                entry_time = time


            elif z_score < -z_threshold:
                pos = -1

                total_pos_size = balance * risk_percent / 100
                sber_pos_size = a / (a + 1) * total_pos_size
                sberp_pos_size = total_pos_size / (a + 1)

                sber_quantity = sber_pos_size // sber_price
                sberp_quantity = sberp_pos_size // sberp_price
                sber_entry_price = sber_price
                sberp_entry_price = sberp_price

                entry_time = time

        else:
            if pos == 1 and z_score <= 0:
                pos = 0

                long_pnl = (sberp_price - sberp_entry_price) * sberp_quantity
                short_pnl = (sber_entry_price - sber_price) * sber_quantity
                total_fee = fee * (sber_quantity * sber_price + sberp_quantity * sberp_price) * 2
                paid_fees += total_fee
                balance += long_pnl + short_pnl - total_fee

                pnls.append(long_pnl + short_pnl)
                holding_times.append(time - entry_time)

            elif pos == -1 and z_score >= 0:
                pos = 0

                long_pnl = (sber_price - sber_entry_price) * sber_quantity
                short_pnl = (sberp_entry_price - sberp_price) * sberp_quantity
                total_fee = fee * (sber_quantity * sber_price + sberp_quantity * sberp_price) * 2
                paid_fees += total_fee
                balance += long_pnl + short_pnl - total_fee

                pnls.append(long_pnl + short_pnl)
                holding_times.append(time - entry_time)

        equity_curve.append(balance)

    days = (timestamps[-1] - timestamps[0]).days
    years = days / 365.25
    annualized_return = ((equity_curve[-1] / initial_balance) ** (1 / years) - 1)*100

    avg_holding_time = sum(holding_times, timedelta(0)) / len(holding_times)
    total_trades = len(pnls)
    max_pnl = max(pnls)
    min_pnl = min(pnls)
    avg_pnl = sum(pnls)/len(pnls)
    winning_trades = sum(1 for x in pnls if x > 0)
    win_ratio = winning_trades/total_trades

    equity_series = pd.Series(equity_curve, index=pd.to_datetime(timestamps))
    returns_10min = equity_series.pct_change().dropna()
    N = 252 * 39
    sharpe = returns_10min.mean() / returns_10min.std() * np.sqrt(N)

    if plot:
        plt.figure(figsize=(12, 5))
        plt.plot(timestamps, equity_curve, linewidth=2)
        plt.title("Equity curve")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()


    return sharpe,balance - initial_balance, (balance - initial_balance)/initial_balance * 100, max_pnl, min_pnl, avg_pnl, win_ratio, total_trades, avg_holding_time, annualized_return, paid_fees

def objective(trial, df):
    df = df.copy()
    z_threshold = trial.suggest_float('z_threshold', 0.5,5)
    z_window = trial.suggest_int('z_window', 5,50)
    spread_window = trial.suggest_int('spread_window', 10,3000, log=True)

    sber_price_arr, sberp_price_arr, z_score_arr, a_arr = prepare_data_arrays(df, z_window, spread_window)

    return run_strategy_fast(sber_price_arr, sberp_price_arr, z_score_arr, a_arr, z_threshold)

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


if __name__ == "__main__":
    df = open_data(timeframe=1)
    windows = generate_walkforward_windows(df)

    results = []
    test_results = []
    for train_start, train_end, test_start, test_end in windows:
        print(f"\nPeriod: {train_start.date()} — {train_end.date()} (train), {test_start.date()} — {test_end.date()} (test)")
        train_df = df.loc[train_start:train_end].copy()
        test_df = df.loc[test_start:test_end].copy()

        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, train_df), n_trials=20, n_jobs=8)

        best_params = study.best_params
        z_threshold = best_params['z_threshold']
        spread_window = best_params['spread_window']
        z_window = best_params['z_window']

        sber_price_arr, sberp_price_arr, z_score_arr, a_arr = prepare_data_arrays(test_df, z_window, spread_window)
        timestamps = test_df.index[-len(sberp_price_arr):]
        sharpe,absolute_profit, profit, max_pnl, min_pnl, avg_pnl, win_ratio, total_trades, avg_holding_time, annualized_return, paid_fees = test_strategy_slow(sber_price_arr, sberp_price_arr, z_score_arr, a_arr, z_threshold, timestamps=timestamps, plot=False)

        print("-----------------------------------")
        print(f"Test profit = {profit:.1f}%")
        print(f"Test absolute profit = {absolute_profit}")
        print(f"Annualized return = {annualized_return:.1f}%")
        print(f"Anuualized sharpe =  {sharpe}")
        print(f"Total trades =  {total_trades}")
        print(f"Max_pnl = {max_pnl:.0f}")
        print(f"Min_pnl = {min_pnl:.0f}")
        print(f"Average pnl = {avg_pnl:.0f}")
        print(f"Win ratio =  {win_ratio*100:.0f}%")
        print(f"Average holding time =  {avg_holding_time.total_seconds() / 3600:.1f} hours")
        print(f"Total paid fees = {paid_fees:.1f}")
        print("Best params: ", best_params)
        test_results.append(annualized_return)

        results.append({
            "train_start": train_start.date(),
            "train_end": train_end.date(),
            "test_start": test_start.date(),
            "test_end": test_end.date(),
            "annualized_return_%": annualized_return,
            "sharpe": sharpe,
            "profit_%": profit,
            "absolute_profit": absolute_profit,
            "total_trades": total_trades,
            "win_ratio_%": win_ratio * 100,
            "avg_holding_hours": avg_holding_time.total_seconds() / 3600,
            "max_pnl": max_pnl,
            "min_pnl": min_pnl,
            "avg_pnl": avg_pnl,
            "paid_fees": paid_fees,
            "z_threshold": z_threshold,
            "spread_window": spread_window,
            "z_window": z_window,
        })


    print("------------------------------")
    print(f"Average annualized return = {sum(test_results)/len(test_results):.1f}%")

    results_df = pd.DataFrame(results)
    results_df.to_csv("walk_forward_results.csv", index=False)
    results_df.to_markdown("walk_forward_results.md", index=False)

    print(results_df)

