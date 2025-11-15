import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS


def prepare_data():
    df_sber = pd.read_csv("data/SBER10min.csv", parse_dates=["timestamp"], index_col="timestamp")
    df_sberp = pd.read_csv("data/SBERP10min.csv", parse_dates=["timestamp"], index_col="timestamp")

    df_sber = df_sber[['close']].copy()
    df_sberp = df_sberp[['close']].copy()

    df_sber.rename(columns={'close': 'SBER'}, inplace=True)
    df_sberp.rename(columns={'close': 'SBERP'}, inplace=True)

    df = pd.concat([df_sber, df_sberp], axis=1).dropna()

    return df

def run_srtategy(df, z_threshold, z_window, spread_window):
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

    balance = 1000000
    risk_percent = 10
    pos = 0
    # SBER = a * SBERP + b
    # d(SBER) = a * d(SBERP)
    # SPREAD = SBER - a * SBERP - b
    # z > 0 - short a * SBER, long SBERP
    # z < 0 - long a * SBER, short SBERP
    for row in df.itertuples():
        sber_price = row.SBER
        sberp_price = row.SBERP
        a = row.a
        z_score = row.z_score

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


data = prepare_data()
print(run_srtategy(data, 1,10,100))