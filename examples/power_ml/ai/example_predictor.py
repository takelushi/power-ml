"""Example of predictor."""

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression

from power_ml.ai.metrics import MAE
from power_ml.ai.predictor.light_gbm import LightGBM
from power_ml.ai.predictor.sklearn_predictor import SklearnPredictor

x, y = load_boston(return_X_y=True)
x_trn, y_trn = x[:400], y[:400]
x_tst, y_tst = x[400:], y[400:]

# Linear Regression
predictor = SklearnPredictor(LinearRegression, 'regression')
predictor.train(x_trn, y_trn)
y_pred = predictor.predict(x_tst)
print(MAE(y_tst, y_pred))

# LightGBM
param = {
    'objective': 'regression',
}
predictor = LightGBM('regression', param=param)
predictor.train(x_trn, y_trn)
y_pred = predictor.predict(x_tst)
print(MAE(y_tst, y_pred))

# Get train parameters.
print(predictor.get_train_param())

# Get information
print(predictor.info)

# Check predictor is same or not with hash.
new_predictor = LightGBM('regression', param=param)
predictor_hash = predictor.info['hash']
new_predictor_hash = new_predictor.hash_train(x_trn, y_tst)
print(predictor_hash == new_predictor_hash)
