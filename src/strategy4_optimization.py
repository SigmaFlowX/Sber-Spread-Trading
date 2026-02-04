import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from numba import njit
from dateutil.relativedelta import relativedelta
import optuna
import os
import matplotlib.pyplot as plt
from datetime import timedelta
import numpy as np
from optuna.visualization import plot_optimization_history

#optuna.logging.set_verbosity(optuna.logging.CRITICAL)   #to hide optuna study logs

STARTING_BALANCE = 100000
INITIAL_POS_SIZE = 10 #%  before there are enough trades to use Kelly criterion
KELLY_N = 50          # window for Kelly criterion
ZERO_KELLY = 0.25    # Kelly when var = 0.0 and pnls are positive
MAX_KELLY = 0.5    # A boundary for some sanity
FEE = 0.008/100
SINCE = "01-01-2024" #None to use all the data
TIMEFRAME = 1  #1 or 10 (min)
N_TRIALS = 100    #optuna study trials
N_TRAIN_MONTHS = 6
N_TEST_MONTHS = 3
PLOT_EQUITIES = False
OPTUNA_VISUALIZE = True

@njit(cache=True)
def calculate_total_pos_size(kelly_count, kelly_pnls, balance):
    if kelly_count >= KELLY_N:
        mean_pnl = np.mean(kelly_pnls)
        var_pnl = np.var(kelly_pnls)
        if var_pnl != 0.0:
            kelly = mean_pnl / var_pnl
            kelly = max(0.0, min(MAX_KELLY, kelly))
        else:
            if mean_pnl > 0:
                kelly = ZERO_KELLY
            else:
                kelly = INITIAL_POS_SIZE / 100
        total_pos_size = balance * kelly
    else:
        total_pos_size = balance * INITIAL_POS_SIZE / 100

    return total_pos_size

def performance_metrics(equity, risk_free_rate=0):
    returns = equity.pct_change().dropna()

    total_return = equity.iloc[-1] / equity.iloc[0]
    n_years = (equity.index[-1] - equity.index[0]).days / 365
    ann_return = total_return ** (1 / n_years) - 1

    trading_hours = 8.75
    bars_per_day = (trading_hours * 60) / TIMEFRAME
    bars_per_year = 252 * bars_per_day

    rf_period = risk_free_rate / bars_per_year
    excess_returns = returns - rf_period
    sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(bars_per_year)

    return ann_return, sharpe

def open_data(timeframe, since=None):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "..", "data")

    df_sber = pd.read_csv(os.path.join(data_dir, f"SBER{timeframe}min.csv"), parse_dates=["timestamp"], index_col="timestamp")
    df_sberp = pd.read_csv(os.path.join(data_dir, f"SBERP{timeframe}min.csv"),parse_dates=["timestamp"], index_col="timestamp")

    df_sber = df_sber[['close', 'open']].copy()
    df_sberp = df_sberp[['close', 'open']].copy()

    df_sber.rename(columns={'close': 'SBER_close', 'open':'SBER_open'}, inplace=True)
    df_sberp.rename(columns={'close': 'SBERP_close', 'open': 'SBERP_open'}, inplace=True)

    df = pd.concat([df_sber, df_sberp], axis=1).dropna()

    if since is not None:
        df = df[df.index >= pd.to_datetime(since)]

    return df

def prepare_data_arrays(df, z_window, spread_window):
    df = df.copy()

    y = df['SBER_close']
    X = sm.add_constant(df['SBERP_close'])

    rols = RollingOLS(y, X, window=spread_window)
    rres = rols.fit()

    params = rres.params
    df['a'] = params['SBERP_close']
    df['b'] = params['const']
    df['spread'] = y - (df['a'] * df['SBERP_close'] + df['b'])

    df['spread_mean'] = df['spread'].rolling(z_window).mean()
    df['spread_std'] = df['spread'].rolling(z_window).std()
    df['z_score'] = (df['spread'] - df['spread_mean']) / df['spread_std']

    df = df.dropna()

    return (df['SBER_close'].values,
            df['SBERP_close'].values,
            df['SBER_open'].values,
            df['SBERP_open'].values,
            df['z_score'].values,
            df['a'].values)

@njit(cache=True)
def run_strategy_fast(sber_close_arr, sberp_close_arr, sber_open_arr, sberp_open_arr, z_score_arr, a_arr, z_entry, z_exit, sl_pct, initial_balance=100000):
    balance = initial_balance
    pos = 0

    kelly_pnls = np.empty(KELLY_N, dtype=np.float64)
    kelly_count = 0

    # SBER = a * SBERP + b
    # d(SBER) = a * d(SBERP)
    # SPREAD = SBER - a * SBERP - b
    # z > 0 - short a * SBER, long SBERP
    # z < 0 - long a * SBER, short SBERP
    for i in range(0, len(sberp_close_arr)-1):
        sber_price = sber_open_arr[i+1]   #i close is used to calculated z_score, the next available price is i+1 open
        sberp_price = sberp_open_arr[i+1]
        a = a_arr[i]
        z_score = z_score_arr[i]

        if pos == 0:
            if z_score > z_entry:
                pos = 1

                total_pos_size = calculate_total_pos_size(kelly_count, kelly_pnls, balance)

                sber_pos_size = a/(a+1) * total_pos_size
                sberp_pos_size = total_pos_size/(a+1)

                sber_quantity = int(sber_pos_size // sber_price)
                sberp_quantity = int(sberp_pos_size // sberp_price)
                sber_entry_price = sber_price
                sberp_entry_price = sberp_price

                entry_balance = balance

            elif z_score < -z_entry:
                pos = -1

                total_pos_size = calculate_total_pos_size(kelly_count, kelly_pnls, balance)

                sber_pos_size = a / (a + 1) * total_pos_size
                sberp_pos_size = total_pos_size / (a + 1)

                sber_quantity = int(sber_pos_size // sber_price)
                sberp_quantity = int(sberp_pos_size // sberp_price)
                sber_entry_price = sber_price
                sberp_entry_price = sberp_price

                entry_balance = balance
        else:
            if pos == 1:
                long_pnl = (sberp_price - sberp_entry_price) * sberp_quantity
                short_pnl = (sber_entry_price - sber_price) * sber_quantity
                total_fee = FEE * (sber_quantity * sber_price + sberp_quantity * sberp_price) * 2
                total_pnl = long_pnl + short_pnl - total_fee

                if total_pnl < -entry_balance * sl_pct / 100 or z_score <= z_exit:
                    balance += total_pnl
                    pos = 0

                    kelly_pnls[kelly_count % KELLY_N] = total_pnl / entry_balance
                    kelly_count += 1

            elif pos == -1:
                long_pnl = (sber_price - sber_entry_price) * sber_quantity
                short_pnl = (sberp_entry_price - sberp_price) * sberp_quantity
                total_fee = FEE * (sber_quantity * sber_price + sberp_quantity * sberp_price) * 2
                total_pnl = long_pnl + short_pnl - total_fee
                if total_pnl < -entry_balance * sl_pct / 100 or z_score >= -z_exit:
                    balance += total_pnl
                    pos = 0

                    kelly_pnls[kelly_count % KELLY_N] = total_pnl / entry_balance
                    kelly_count += 1


    return (balance - initial_balance)/initial_balance * 100

def test_strategy_slow(sber_close_arr, sberp_close_arr, sber_open_arr, sberp_open_arr,  z_score_arr, a_arr, z_entry, z_exit, sl_pct, timestamps, initial_balance=100000,plot=False):
    balance = initial_balance
    pos = 0
    pnls = []
    equity_curve = []
    holding_times = []
    paid_fees = 0
    kelly_pnls = np.empty(KELLY_N, dtype=np.float64)
    kelly_count = 0

    for i in range(0, len(sberp_close_arr)-1):
        sber_price = sber_open_arr[i+1]
        sberp_price = sberp_open_arr[i+1]
        a = a_arr[i]
        z_score = z_score_arr[i] # no look-ahead bias
        time = timestamps[i]

        if pos == 0:
            if z_score > z_entry:
                pos = 1

                total_pos_size = calculate_total_pos_size(kelly_count, kelly_pnls, balance)
                sber_pos_size = a/(a+1) * total_pos_size
                sberp_pos_size = total_pos_size/(a+1)

                sber_quantity = int(sber_pos_size // sber_price)
                sberp_quantity = int(sberp_pos_size // sberp_price)
                sber_entry_price = sber_price
                sberp_entry_price = sberp_price

                entry_balance = balance

                entry_time = time


            elif z_score < -z_entry:
                pos = -1

                total_pos_size = calculate_total_pos_size(kelly_count, kelly_pnls, balance)
                sber_pos_size = a / (a + 1) * total_pos_size
                sberp_pos_size = total_pos_size / (a + 1)

                sber_quantity = int(sber_pos_size // sber_price)
                sberp_quantity = int(sberp_pos_size // sberp_price)
                sber_entry_price = sber_price
                sberp_entry_price = sberp_price

                entry_balance = balance

                entry_time = time

        else:
            if pos == 1:
                long_pnl = (sberp_price - sberp_entry_price) * sberp_quantity
                short_pnl = (sber_entry_price - sber_price) * sber_quantity
                total_fee = FEE * (sber_quantity * sber_price + sberp_quantity * sberp_price) * 2
                total_pnl = long_pnl + short_pnl - total_fee
                if total_pnl < - entry_balance * sl_pct / 100 or z_score <= z_exit:
                    paid_fees += total_fee
                    balance += total_pnl
                    pnls.append(long_pnl + short_pnl)
                    holding_times.append(time - entry_time)
                    pos = 0

                    kelly_pnls[kelly_count % KELLY_N] = total_pnl / entry_balance
                    kelly_count += 1

            elif pos == -1:
                long_pnl = (sber_price - sber_entry_price) * sber_quantity
                short_pnl = (sberp_entry_price - sberp_price) * sberp_quantity
                total_fee = FEE * (sber_quantity * sber_price + sberp_quantity * sberp_price) * 2
                total_pnl = long_pnl + short_pnl - total_fee
                if total_pnl < - entry_balance * sl_pct / 100 or z_score >= -z_exit:
                    paid_fees += total_fee
                    balance += total_pnl

                    pnls.append(long_pnl + short_pnl)
                    holding_times.append(time - entry_time)
                    pos = 0

                    kelly_pnls[kelly_count % KELLY_N] = total_pnl / entry_balance
                    kelly_count += 1

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
    equity_series = pd.Series(equity_curve, index=pd.to_datetime(timestamps[:len(equity_curve)]))

    if plot:
        plt.figure(figsize=(12, 5))
        plt.plot(timestamps, equity_curve, linewidth=2)
        plt.title("Equity curve")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()


    return balance, equity_series, balance - initial_balance, (balance - initial_balance)/initial_balance * 100, max_pnl, min_pnl, avg_pnl, win_ratio, total_trades, avg_holding_time, annualized_return, paid_fees

def objective(trial, df):
    df = df.copy()
    z_entry = trial.suggest_float('z_entry', 0.0,5)
    z_exit = trial.suggest_float('z_exit', 0.0, z_entry)
    sl_pct = trial.suggest_float('sl_pct', 1,50)
    z_window = trial.suggest_int('z_window', 5,10000, log=True)
    spread_window = trial.suggest_int('spread_window', 10,10000, log=True)

    sber_close, sberp_close,sber_open, sberp_open, z_score_arr, a_arr = prepare_data_arrays(df, z_window, spread_window)

    return run_strategy_fast(sber_close, sberp_close, sber_open, sberp_open, z_score_arr, a_arr, z_entry, z_exit, sl_pct, STARTING_BALANCE)

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
        current_start = train_start + relativedelta(months=test_months)

    return windows


if __name__ == "__main__":
    df = open_data(timeframe=TIMEFRAME, since=SINCE)
    windows = generate_walkforward_windows(df, train_months=N_TRAIN_MONTHS, test_months=N_TEST_MONTHS)

    balance = STARTING_BALANCE

    results = []
    equity_series = []
    for train_start, train_end, test_start, test_end in windows:
        print(f"\nPeriod: {train_start.date()} — {train_end.date()} (train), {test_start.date()} — {test_end.date()} (test)")
        train_df = df.loc[train_start:train_end].copy()
        test_df = df.loc[test_start:test_end].copy()

        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, train_df), n_trials=N_TRIALS, n_jobs=8)

        if OPTUNA_VISUALIZE:
            fig = plot_optimization_history(study)
            fig.show()

        best_params = study.best_params
        z_entry = best_params['z_entry']
        z_exit = best_params['z_exit']
        sl_pct = best_params['sl_pct']
        spread_window = best_params['spread_window']
        z_window = best_params['z_window']

        sber_close, sberp_close, sber_open, sberp_open, z_score_arr, a_arr = prepare_data_arrays(test_df, z_window, spread_window)
        timestamps = test_df.index[-len(sberp_close):]

        (
            final_balance,
            equity,
            absolute_profit,
            profit,
            max_pnl,
            min_pnl,
            avg_pnl,
            win_ratio,
            total_trades,
            avg_holding_time,
            annualized_return,
            paid_fees,
        ) = test_strategy_slow(
            sber_close,
            sberp_close,
            sber_open,
            sberp_open,
            z_score_arr,
            a_arr,
            z_entry,
            z_exit,
            sl_pct,
            timestamps=timestamps,
            initial_balance=balance,
            plot=PLOT_EQUITIES,
        )

        equity_series.append(equity)
        balance = final_balance

        print(f"Test profit = {profit:.1f}%")
        print(f"Test absolute profit = {absolute_profit}")
        print(f"Annualized return = {annualized_return:.1f}%")
        print(f"Total trades =  {total_trades}")
        print(f"Max_pnl = {max_pnl:.0f}")
        print(f"Min_pnl = {min_pnl:.0f}")
        print(f"Average pnl = {avg_pnl:.0f}")
        print(f"Win ratio =  {win_ratio*100:.0f}%")
        print(f"Average holding time =  {avg_holding_time.total_seconds() / 3600:.1f} hours")
        print(f"Total paid fees = {paid_fees:.1f}")
        print("Best params: ", best_params)


        print("-----------------------------------")

        results.append({
            "train_start": train_start.date(),
            "train_end": train_end.date(),
            "test_start": test_start.date(),
            "test_end": test_end.date(),
            "annualized_return_%": round(annualized_return, 1),
            "profit_%": round(profit, 1),
            "absolute_profit": round(absolute_profit, 0),
            "total_trades": total_trades,
            "win_ratio_%": round(win_ratio * 100, 1),
            "avg_holding_hours": round(avg_holding_time.total_seconds() / 3600,1),
            "max_pnl": round(max_pnl, 1),
            "min_pnl": round(min_pnl, 1),
            "avg_pnl": round(avg_pnl,1),
            "paid_fees": round(paid_fees,1),
            "z_entry": z_entry,
            "z_exit": z_exit,
            "sl_pct": round(sl_pct, 1),
            "spread_window": spread_window,
            "z_window": z_window,
        })


    print("------------------------------")

    total_equity = pd.concat(equity_series).sort_index()
    ann_return, sharpe = performance_metrics(total_equity)

    print(f"Total return since {SINCE} is {(balance - STARTING_BALANCE)/STARTING_BALANCE * 100:.1f} %")
    print(f"Annualized return is {ann_return*100:.2f}%")
    print(f"Annualized sharpe ratio is {sharpe:.1f}")

    plt.figure(figsize=(12, 5))
    plt.plot(total_equity.index, total_equity)
    plt.title("Equity curve")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()



    results_df = pd.DataFrame(results)

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    results_dir = os.path.join(base_dir, "results")

    results_df.to_csv(os.path.join(results_dir, "walk_forward_results.csv"), index=False)
    results_df.to_markdown(os.path.join(results_dir, "walk_forward_results.md"), index=False)

    print(results_df)

