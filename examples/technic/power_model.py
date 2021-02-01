"""Example of Power Model."""

import pandas as pd
from sklearn.datasets import load_boston

from power_ml.ai.predictor.light_gbm import LightGBM
from power_ml.data.local_pickle_store import LocalPickleStore
from power_ml.platform.power_model import PowerModel

store = LocalPickleStore('./tmp/store')

dataset = load_boston()
data = pd.DataFrame(dataset['data'], columns=dataset['feature_names'])
data['target'] = dataset['target']

predictor_class = LightGBM
train_param = {'objective': 'regression'}

power_model = PowerModel('regression', 'target', store, data, seed=2)
print(power_model.calc_column_stats())

base_idx = power_model.split_trn_val(0.8)
print('Train index: {}'.format(base_idx[0]))
base_model_id = power_model.train(predictor_class,
                                  base_idx[0],
                                  param=train_param)
print('model: {}'.format(base_model_id))

print('Validation index: {}'.format(base_idx[1]))
y_pred_id = power_model.predict(base_model_id, base_idx[1])
print('Y pred: {}'.format(y_pred_id))

score = power_model.validate(base_model_id, base_idx[1])[0]
print('Score: {}'.format(score))

val_idx_li = power_model.create_validation_index('seed', {
    'seeds': [1, 2, 3, 4],
    'trn_ratio': 0.8,
})
score, _ = power_model.validate_each(predictor_class,
                                     val_idx_li,
                                     train_param=train_param)
print(score)

trn_perms = power_model.calc_perm(base_model_id, base_idx[0], 10)
print(trn_perms['MAE'])

val_perms = power_model.calc_perm(base_model_id, base_idx[1], 10)
print(val_perms['MAE'])
