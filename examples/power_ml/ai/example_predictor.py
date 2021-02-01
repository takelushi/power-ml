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
predictor.train(trn_x, trn_y)
score, _ = predictor.evaluate(tst_x, tst_y)
print(score)

# LightGBM
param = {
    'objective': 'regression',
}
predictor = LightGBM('regression', param=param)
predictor.train(trn_x, trn_y)
score, _ = predictor.evaluate(tst_x, tst_y)
print(score)

# Get train parameters.
print(predictor.get_train_param())

# Get information
print(predictor.info)

# Check predictor is same or not with hash.
new_predictor = LightGBM('regression', param=param)
predictor_hash = predictor.info['hash']
new_predictor_hash = new_predictor.hash_train(trn_x, trn_y)
print(predictor_hash == new_predictor_hash)
