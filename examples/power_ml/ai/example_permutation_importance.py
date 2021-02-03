"""Example of Permutation Importance."""

import pandas as pd
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression

from power_ml.ai.metrics import MAE
from power_ml.ai.permutation_importance import PermutationImportance
from power_ml.ai.predictor.sklearn_predictor import SklearnPredictor
from power_ml.util.seed import set_seed

dataset = load_boston()
x = pd.DataFrame(dataset['data'], columns=dataset['feature_names'])
y = dataset['target']

# Train model.
predictor = SklearnPredictor(LinearRegression, 'regression')
predictor.train(x, y)

# Initialize.
perm = PermutationImportance(predictor, MAE, x, y, n=20)

# Recommend to set seed.
set_seed(3)

# Get single column shuffled metric.
col = 'DIS'
metric = perm.shuffle_and_evaluate(col)
print('{}: {}'.format(col, metric))

# Iterate permutation importance,
for col_perm in perm.iter_perm():
    col, weight, score = col_perm
    print('{:10},  {:.4f},  {:.4f}'.format(col, weight, score))

# Analyzed permutation importance
# n_jobs=-1: Use all CPU cores.
perms = perm.calc(n_jobs=-1)
print(perms)

# Check MAE.
print(MAE(y, predictor.predict(x)))

# Drop worse features.
worse_features = list(perms[perms['Type'] == 'worse']['Column'])
print('Worse features: {}'.format(worse_features))
x = x.drop(columns=worse_features)
redictor = SklearnPredictor(LinearRegression, 'regression')
predictor.train(x, y)

# MAE will more better.
print(MAE(y, predictor.predict(x)))
