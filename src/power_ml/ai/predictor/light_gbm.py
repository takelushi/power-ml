"""LightGBM."""

import lightgbm as lgb
import numpy as np

from power_ml.ai.predictor.base_predictor import BasePredictor
from power_ml.ai.types import X_TYPE, Y_TYPE


class LightGBM(BasePredictor):
    """Sckit-learn Predictor."""

    def get_train_param(self) -> dict:
        """Get fit parameter."""
        return self._model.get_params()

    def init(self, param: dict) -> None:
        """Initialize predictor."""
        self._model = lgb.LGBMModel(**param)

    def _train(self, x: X_TYPE, y: Y_TYPE):
        """Fit."""
        self._model.fit(x, y)

    def _predict(self, x: X_TYPE, param: dict = None) -> np.ndarray:
        """Predict."""
        if param is None:
            param = {}
        return self._model.predict(x, **param)
