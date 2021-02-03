"""Base Predictor."""

from abc import ABC, abstractmethod

import numpy as np

from power_ml.ai.types import X_TYPE, Y_TYPE


class BasePredictor(ABC):
    """Base Predictor."""

    def __init__(self,
                 target_type: str,
                 param: dict = None,
                 meta: dict = None) -> None:
        """Initialize object."""
        self.name = self.__class__.__name__
        self.target_type = target_type
        self.meta = {} if meta is None else meta

        self._train_param = {} if param is None else param
        self.init(self._train_param)

    @abstractmethod
    def init(self, param: dict) -> None:
        """Initialize predictor."""
        raise NotImplementedError()

    def get_train_param(self) -> dict:
        """Get fit parameter."""
        return self._train_param

    @abstractmethod
    def _train(self, x: X_TYPE, y: Y_TYPE):
        """Fit."""
        raise NotImplementedError()

    def train(self, x: X_TYPE, y: Y_TYPE):
        """Fit."""
        self._train(x, y)

    @abstractmethod
    def _predict(self, x: X_TYPE, param: dict) -> np.ndarray:
        """Predict."""
        raise NotImplementedError()

    def predict(self, x: X_TYPE, param: dict = None) -> np.ndarray:
        """Predict."""
        if param is None:
            param = {}
        y = self._predict(x, param)
        return y
