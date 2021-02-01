"""Model."""

from typing import Any, Type

import joblib
import numpy as np
import pandas as pd

from power_ml.ai.metrics import BaseMetric, METRICS, NumericMetric
from power_ml.ai.permutation_importance import PermutationImportance
from power_ml.ai.predictor.base_predictor import BasePredictor
from power_ml.ai.types import X_TYPE, Y_TYPE
from power_ml.util import dict_util


class Model:
    """Model."""

    def __init__(self,
                 predictor: BasePredictor,
                 metrics: list[Type[BaseMetric]] = None) -> None:
        """Initialize object."""
        self._predictor = predictor  # type: ignore
        self.info: dict[str, Any] = {
            'predictor': {
                'target_type': self._predictor.target_type,
                'class': self._predictor.__class__.__name__,
            },
        }
        self.perms: dict[str, pd.DataFrame] = {}

        if metrics is None:
            metrics = list(METRICS[self._predictor.target_type].values())
        self.metrics = metrics

    def hash_model(self, x: X_TYPE, y: Y_TYPE) -> str:
        """Hash train."""
        self.info['train'] = {
            'param': self._predictor.get_train_param(),
            'x_hash': joblib.hash(x),
            'y_hash': joblib.hash(y),
        }
        d = {
            'predictor': self.info['predictor'],
            'train': self.info['train'],
        }

        return dict_util.to_hash(d)

    def train(self, x: X_TYPE, y: Y_TYPE) -> None:
        """Train model."""
        self.info['id'] = self.hash_model(x, y)

        self._predictor.train(x, y)
        self.info['train_metrics'] = self.validate(x, y)

    def predict(self, x: X_TYPE) -> np.ndarray:
        """Predict model."""
        y_pred = self._predictor.predict(x)
        return y_pred

    def validate(self,
                 x: X_TYPE,
                 y_true: Y_TYPE,
                 param: dict = None) -> tuple[dict, np.ndarray]:
        """Validate."""
        y_pred = self._predictor.predict(x, param=param)

        result = {}
        for metric_class in self.metrics:
            name = metric_class.__name__
            result[name] = metric_class(y_true, y_pred)

        return result, y_pred

    def calc_perm(self,
                  x: pd.DataFrame,
                  y: Y_TYPE,
                  metrics: list[Type[NumericMetric]] = None,
                  n: int = 1,
                  n_jobs: int = 1) -> dict[str, pd.DataFrame]:
        if metrics is None:
            metrics = self.metrics

        for metric in metrics:  # type: ignore
            perm = PermutationImportance(self._predictor, metric, x, y, n=n)
            perms = perm.calc(n_jobs=n_jobs)
            self.perms[metric.__name__] = perms
        return self.perms
