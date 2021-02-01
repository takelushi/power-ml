"""Example of project."""

from sklearn.datasets import load_boston

from power_ml.ai.metrics import MAE
from power_ml.ai.predictor.light_gbm import LightGBM
from power_ml.data.local_pickle_store import LocalPickleStore
from power_ml.platform.project import Project

store = LocalPickleStore('./tmp/store')

project = Project('regression', [MAE], store)

x, y = load_boston(return_X_y=True)
x_trn_name = store.save(x[:400])
y_trn_name = store.save(y[:400])
x_tst_name = store.save(x[400:])
y_tst_name = store.save(y[400:])

param = {
    'objective': 'regression',
}

predictor_hash = project.train(LightGBM, param, x_trn_name, y_trn_name)
score = project.evaluate(predictor_hash, x_tst_name, y_tst_name)

print(score)
print(project.models)
