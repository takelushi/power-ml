"""Parallel ml."""

import combu
from sklearn.datasets import load_boston

from power_ml.ai.predictor.light_gbm import LightGBM
from power_ml.data.local_pickle_store import LocalPickleStore

store = LocalPickleStore('./tmp/store')

x, y = load_boston(return_X_y=True)
trn_x_name = store.save(x[:400])
trn_y_name = store.save(y[:400])
tst_x_name = store.save(x[400:])
tst_y_name = store.save(y[400:])


def train(param: dict) -> tuple[LightGBM, dict]:
    """Train function."""
    trn_x = store.load(trn_x_name)
    trn_y = store.load(trn_y_name)
    tst_x = store.load(tst_x_name)
    tst_y = store.load(tst_y_name)

    predictor = LightGBM('regression')
    predictor.fit(trn_x, trn_y, param=param)

    score, _ = predictor.evaluate(tst_x, tst_y)

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
