"""LightGBM."""

import lightgbm as lgb
import numpy as np

from power_ml.ai.predictor.base_predictor import BasePredictor
from power_ml.ai.types import X_TYPE, Y_TYPE


class LightGBM(BasePredictor):
    """Sckit-learn Predictor."""

    def fit(self, x: X_TYPE, y: Y_TYPE, param: dict = None):
        """Fit."""
        if param is None:
            param = {}

        self._model = lgb.LGBMModel(**param)
        self._model.fit(x, y)

    def predict(self, x: X_TYPE, param: dict = None) -> np.ndarray:
        """Predict."""
        if param is None:
            param = {}
        return self._model.predict(x, **param)
