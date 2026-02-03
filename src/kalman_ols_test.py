from strategy2_optimization import open_data
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from pykalman import KalmanFilter
import numpy as np


def prepare_data(df, z_window, spread_window):
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

    return df

df_original = open_data(timeframe='10', since=None)


y = df_original['SBER'].values
x = df_original['SBERP'].values
H = np.column_stack([x, np.ones(len(x))])
H = H[:, np.newaxis, :]

kf = KalmanFilter(
    transition_matrices=np.eye(2),
    observation_matrices=H,
    observation_covariance=1.0,
    transition_covariance=0.01 * np.eye(2),
    initial_state_mean=np.zeros(2),
    initial_state_covariance=np.eye(2)
)

state_means, _ = kf.filter(y)
beta = state_means[:, 0]
intercept = state_means[:, 1]

df_params = pd.DataFrame(
    {
        "beta": beta,
        "intercept": intercept,
    },
    index=df_original.index
)

plt.figure(figsize=(12, 5))
plt.plot(df_params.index, df_params["beta"], label="Kalman beta")


N = 10
start = 10000
end = 100000

step = (end - start)/(N-1)

for i in range(N):
    window = int(start + i * step)
    print(window)
    df = prepare_data(df_original, 10, window)
    plt.plot(df.index, df['a'], label=str(window), alpha=0.7)


plt.title("Kalman beta vs rolling OLS")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()