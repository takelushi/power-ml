"""Base Predictor."""

from abc import ABC, abstractmethod
from typing import Type

import numpy as np

from power_ml.ai.metrics import BaseMetric, METRICS
from power_ml.ai.types import X_TYPE, Y_TYPE


class BasePredictor(ABC):
    """Base Predictor."""

    def __init__(self,
                 target_type: str,
                 metrics: list[Type[BaseMetric]] = None,
                 meta: dict = None) -> None:
        """Initialize object."""
        self.name = self.__class__.__name__
        self.target_type = target_type
        if metrics is None:
            metrics = list(METRICS[self.target_type].values())
        self.metrics = metrics
        self.meta = {} if meta is None else meta

    @abstractmethod
    def fit(self, x: X_TYPE, y: Y_TYPE, param: dict = None):
        """Fit."""
        raise NotImplementedError()

    @abstractmethod
    def predict(self, x: X_TYPE, param: dict = None) -> np.ndarray:
        """Predict."""
        raise NotImplementedError()

    def evaluate(self,
                 x: X_TYPE,
                 y_true: Y_TYPE,
                 param: dict = None) -> tuple[dict, np.ndarray]:
        """Evaluate."""
        y_pred = self.predict(x, param=param)

        result = {}
        for metric_class in self.metrics:
            name = metric_class.__name__
            result[name] = metric_class(y_true, y_pred)

        return result, y_pred
