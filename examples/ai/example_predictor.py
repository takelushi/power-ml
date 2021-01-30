"""Example of predictor."""

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression

from power_ml.ai.predictor.light_gbm import LightGBM
from power_ml.ai.predictor.sklearn_predictor import SklearnPredictor

x, y = load_boston(return_X_y=True)
trn_x, trn_y = x[:400], y[:400]
tst_x, tst_y = x[400:], y[400:]

# Linear Regression
predictor = SklearnPredictor(LinearRegression, 'regression')
predictor.fit(trn_x, trn_y)
score, _ = predictor.evaluate(tst_x, tst_y)
print(score)

# LightGBM
predictor = LightGBM('regression')
param = {
    'objective': 'regression',
}
predictor.fit(trn_x, trn_y, param=param)
score, _ = predictor.evaluate(tst_x, tst_y)
print(score)
