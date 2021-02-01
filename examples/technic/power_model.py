"""Example of Power Model."""

import pandas as pd
from sklearn.datasets import load_boston

from power_ml.ai.predictor.light_gbm import LightGBM
from power_ml.data.local_pickle_store import LocalPickleStore
from power_ml.platform.catalog import Catalog
from power_ml.platform.power_model import PowerModel

catalog = Catalog('./tmp/catalog.json', LocalPickleStore('./tmp/store'))
dataset = load_boston()
data = pd.DataFrame(dataset['data'], columns=dataset['feature_names'])
data['target'] = dataset['target']

predictor_class = LightGBM
train_param = {'objective': 'regression'}

power_model = PowerModel('regression',
                         LightGBM,
                         'target',
                         catalog,
                         data,
                         train_param=train_param,
                         seed=2)
print(power_model.run_flow('calc_column_stats'))

base_idx = power_model.split_trn_val(0.8)
print('Train index: {}'.format(base_idx[0]))
base_model_id = power_model.run_flow('train')

print('model: {}'.format(base_model_id))

print('Validation index: {}'.format(base_idx[1]))
y_pred_id = power_model.run_flow('predict')

print('Y pred: {}'.format(y_pred_id))
score = power_model.run_flow('validate')[0]
print('Score: {}'.format(score))

val_idx_li = power_model.create_validation_index('seed', {
    'seeds': [1, 2, 3, 4],
    'trn_ratio': 0.8,
})
score, _ = power_model.run_flow('validate_each')
print(score)

trn_perms = power_model.run_flow('calc_perm', mode='train')
print(trn_perms['MAE'])
val_perms = power_model.run_flow('calc_perm', mode='validation')
print(val_perms['MAE'])

cols = val_perms['MAE']
d = {}
for row in val_perms['MAE'].itertuples():
    d[row[1]] = row[3]

print(d)

res = power_model.run_flow('calc_perm_each', mode='train')

print(res)
# val_perms, _ = power_model.run_flow('calc_perm_each', mode='validation')
# print(val_perms['MAE'])
