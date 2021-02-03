"""Parallel ml."""

import combu
from sklearn.datasets import load_boston

from power_ml.ai.predictor.light_gbm import LightGBM
from power_ml.data.local_pickle_store import LocalPickleStore

store = LocalPickleStore('./tmp/store')

x, y = load_boston(return_X_y=True)
x_trn_name = store.save(x[:400])
y_trn_name = store.save(y[:400])
x_tst_name = store.save(x[400:])
y_tst_name = store.save(y[400:])


def train(param: dict) -> tuple[LightGBM, dict]:
    """Train function."""
    x_trn = store.load(x_trn_name)
    y_trn = store.load(y_trn_name)
    x_tst = store.load(x_tst_name)
    y_tst = store.load(y_tst_name)

    predictor = LightGBM('regression', param=param)
    predictor.train(x_trn, y_trn)

    score, _ = predictor.evaluate(x_tst, y_tst)

    return predictor, score


param = {
    'objective': ['regression'],
    # 'learning_rate ': [0.01, 0.1, 0.5],
    'feature_fraction': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [None],
    'n_jobs': [1],
}
params = {'param': combu.create_values(param)}

results = []
for res, param in combu.execute(train, params, n_jobs=-1):
    predictor, score = res
    print(score, param)
    results.append({
        'predictor': predictor,
        'score': score,
        'param': param,
    })
