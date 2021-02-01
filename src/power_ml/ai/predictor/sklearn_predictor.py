"""Sckit-learn common predictor."""

from typing import Type

import numpy as np

from power_ml.ai.metrics import BaseMetric
from power_ml.ai.predictor.base_predictor import BasePredictor
from power_ml.ai.types import X_TYPE, Y_TYPE


class SklearnPredictor(BasePredictor):
    """Sckit-learn Predictor."""

    def __init__(self,
                 sklearn_class: type,
                 target_type: str,
                 metrics: list[Type[BaseMetric]] = None,
                 meta: dict = None) -> None:
        """Initialize object."""
        self.sklearn_class = sklearn_class
        super().__init__(target_type, metrics=metrics, meta=meta)

    def get_train_param(self) -> dict:
        """Get fit parameter."""
        return self._model.get_params()

    def init(self, param: dict) -> None:
        """Initialize predictor."""
        self._model = self.sklearn_class(**param)

    def _train(self, x: X_TYPE, y: Y_TYPE):
        """Fit."""
        self._model.fit(x, y)

    def _predict(self, x: X_TYPE, param: dict = None) -> np.ndarray:
        """Predict."""
        if param is None:
            param = {}
        return self._model.predict(x, **param)
