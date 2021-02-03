"""Example of model."""

import pandas as pd
from sklearn.datasets import load_boston

from power_ml.ai.metrics import MAE, MAPE
from power_ml.ai.model import Model
from power_ml.ai.predictor.light_gbm import LightGBM
from power_ml.util.seed import set_seed

# Recommend to set seed.
set_seed(82)

dataset = load_boston()
x = pd.DataFrame(dataset['data'], columns=dataset['feature_names'])
y = dataset['target']
x_trn, y_trn = x[:400], y[:400]
x_tst, y_tst = x[400:], y[400:]

param = {
    'objective': 'regression',
}
predictor = LightGBM('regression', param=param)
model = Model(predictor, metrics=[MAE, MAPE])
model.train(x_trn, y_trn)

# Permutation Importance
perms = model.calc_perm(x_trn, y_trn, n=5, n_jobs=1)
for metric, perm in model.calc_perm(x_trn, y_trn, n=10, n_jobs=1).items():
    print(metric)
    print(perm)
# Also perms ware saved to model.perms.

# Evaluate Score.
score, _ = model.evaluate(x_tst, y_tst)
print(score)
