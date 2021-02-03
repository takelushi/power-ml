"""Example of Cross validation."""

import pandas as pd
from sklearn.datasets import load_boston

from power_ml.ai.metrics import MAE
from power_ml.ai.predictor.light_gbm import LightGBM
from power_ml.ai.validate.cv_generator import CVGenerator
from power_ml.data.local_pickle_store import LocalPickleStore
from power_ml.platform.project import Project

store = LocalPickleStore('./tmp/store')

dataset = load_boston()
x = pd.DataFrame(dataset['data'], columns=dataset['feature_names'])
y = pd.Series(dataset['target'])
x_name = store.save(x)
y_name = store.save(y)
del x
del y

project = Project('regression', [MAE], store)

cv_generator = CVGenerator('KFold', store)
cv_names_li = []
for res in cv_generator.generate(5, x_name, y_name):
    print(res)
    cv_names_li.append(res)

param = {
    'objective': 'regression',
}
score_li = []
for cv_names in cv_names_li:
    predictor_hash = project.train(LightGBM, param, cv_names[0], cv_names[1])
    score = project.evaluate(predictor_hash, cv_names[2], cv_names[3])
    print(score)
    score_li.append(score['MAE'])

score_mean = sum(score_li) / len(score_li)
print('Mean: {}'.format(score_mean))
