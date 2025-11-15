import pandas as pd

def prepare_data():
    df_sber = pd.read_csv("data/SBER10min.csv", parse_dates=["timestamp"], index_col="timestamp")
    df_sberp = pd.read_csv("data/SBERP10min.csv", parse_dates=["timestamp"], index_col="timestamp")

    df_sber = df_sber[['close']].copy()
    df_sberp = df_sberp[['close']].copy()

    df_sber.rename(columns={'close': 'SBER'}, inplace=True)
    df_sberp.rename(columns={'close': 'SBERP'}, inplace=True)

    df = pd.concat([df_sber, df_sberp], axis=1).dropna()

    return df


