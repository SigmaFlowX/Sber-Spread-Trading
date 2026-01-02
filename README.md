# SBER/SBERP Spread Trading Strategy

A simple linear model is used:
Price(SBER) = a * Price(SBERP) + b + E, <br>
where E is stationary and mean revearting.

Then the Z-score is calculated for E = a * Price(SBER) - Price(SBERP) - b <br>

Entry conditions: <br>
If z > z_threshold: short a * SBER and long SBERP <br>
If z < -z_threshold: short SBERP and long a * SBERP

Exit conditions: <br>
z <= 0 <br>
z >= 0 <br>

Walk forward optimization is performed using optuna. <br>
Fees and included in the model, but not the slippage.

## Results
Risking 10% of the initial balance strategy generates on average 6.7% annual return

Sharpe ratio varies in the range of 0.5-2 on 3 month test windows

Win ratio is around 70-85%

Stop losses do not make strategy any better, their imprortance is only about 2%

Main limiting factors are fees and possible slippage (slippage was not modeled)

## Data links

SBER<br>https://drive.google.com/file/d/1HjTwX0ZIwoqtYKKdl5y3fG3rUgcYqZiT/view?usp=sharing

SBERP<br>https://drive.google.com/file/d/1h9tjGM8Yg-Q_uYlmIJrtxOePQApaBU4L/view?usp=sharing

