"""Example of Model Flow."""

import pandas as pd
from sklearn.datasets import load_boston

from power_ml.ai.metrics import MAE, MAPE
from power_ml.ai.predictor.light_gbm import LightGBM
from power_ml.data.local_pickle_store import LocalPickleStore
from power_ml.platform.catalog import Catalog
from power_ml.platform.model_flow import ModelFlow

target_type = 'regression'
target_name = 'target'
predictor_class = LightGBM
train_param = {'objective': 'regression'}
catalog = Catalog('./tmp/catalog.json', LocalPickleStore('./tmp/store'))
metrics = [MAE, MAPE]

dataset = load_boston()
data = pd.DataFrame(dataset['data'], columns=dataset['feature_names'])
data[target_name] = dataset['target']

data_id = catalog.save_table(data)

model_flow = ModelFlow(target_type,
                       predictor_class,
                       target_name,
                       catalog,
                       data_id,
                       metrics=metrics,
                       train_param=train_param,
                       seed=20)

column_stats = model_flow.get_column_stats()
print(column_stats)

part_idx = model_flow.create_partition('random', trn_ratio=0.8)
print('partition index id: {}'.format(part_idx))

if True:
    model_id, score = model_flow.train_validate()
    print('model id: {}'.format(model_id))
    print('score: {}'.format(score))
else:
    model_id = model_flow.train()
    print('model id: {}'.format(model_id))

    score = model_flow.validate()
    print('score: {}'.format(score))

cv_names = model_flow.create_cv_partition('sklearn',
                                          selection='KFold',
                                          n_splits=5)
print('cv partition index id:')
for cv_name in cv_names:
    print('    {}'.format(cv_name))

# CV
if True:
    cv_model_ids, cv_score = model_flow.cross_validation()
    print('cv model id:')
    for model_id in cv_model_ids:
        print('    {}'.format(model_id))
    print('cv mean score (train): {}'.format(cv_score['cv_train']))
    print('cv mean score (validation): {}'.format(cv_score['cv_validation']))
    print('cv each score (train):')
    for score in cv_score['cv_train_each']:
        print('    {}'.format(score))
    print('cv each score (validation):')
    for score in cv_score['cv_validation_each']:
        print('    {}'.format(score))

else:
    cv_model_ids = model_flow.train_cv()
    print('cv model id:')
    for model_id in cv_model_ids:
        print('    {}'.format(model_id))

    cv_score = model_flow.validate_cv()
    print('cv mean score (train): {}'.format(cv_score['cv_train']))
    print('cv mean score (validation): {}'.format(cv_score['cv_validation']))
    print('cv each score (train):')
    for score in cv_score['cv_train_each']:
        print('    {}'.format(score))
    print('cv each score (validation):')
    for score in cv_score['cv_validation_each']:
        print('    {}'.format(score))

# Permutation Importance
perms = model_flow.calc_perm(n=10)
print(perms['train']['MAE'])
print(perms['validation']['MAE'])

# CV Permutation Importance
cv_perms = model_flow.calc_cv_perm(n=10)
print(cv_perms.keys())
print(cv_perms['cv_train'])
