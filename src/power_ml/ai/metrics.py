"""Metrics."""

from abc import ABC, abstractclassmethod
from enum import Enum
from typing import Any, Type

import sklearn.metrics

from power_ml.ai.types import Y_TYPE


class BaseMetric(ABC):
    """Metric."""

    def __new__(cls, y_true: Y_TYPE, y_pred: Y_TYPE, **kwargs: Any) -> Any:
        """Calculate metric."""
        return cls.calc(y_true, y_pred, **kwargs)

    @abstractclassmethod
    def calc(
            cls,  # noqa: N805
            y_true: Y_TYPE,
            y_pred: Y_TYPE,
            **kwargs: Any) -> Any:
        """Calculate metric.

        Args:
            y_true (Y_TYPE): True Y.
            y_pred (Y_TYPE): Predicted Y.

        Returns:
            Any: Result.
        """
        raise NotImplementedError()


class NumericMetric(BaseMetric, ABC):
    """Numeric metric."""

    def __new__(  # type: ignore
            cls, y_true: Y_TYPE, y_pred: Y_TYPE, **kwargs: Any) -> float:
        """Calculate metric."""
        return super().__new__(cls, y_true, y_pred, **kwargs)

    @abstractclassmethod
    def calc(
            cls,  # noqa: N805
            y_true: Y_TYPE,
            y_pred: Y_TYPE,
            **kwargs: Any) -> float:
        """Calculate metric.

        Args:
            y_true (Y_TYPE): True Y.
            y_pred (Y_TYPE): Predicted Y.

        Returns:
            float: Result.
        """
        raise NotImplementedError()


class MAE(NumericMetric):
    """Mean Absolute Error."""

    @classmethod
    def calc(cls, y_true: Y_TYPE, y_pred: Y_TYPE, **kwargs: Any) -> float:
        """Calculate metric."""
        score: float = sklearn.metrics.mean_absolute_error(y_true, y_pred)
        return score


class MAPE(NumericMetric):
    """Mean Absolute Percent Error."""

    @classmethod
    def calc(cls, y_true: Y_TYPE, y_pred: Y_TYPE, **kwargs: Any) -> float:
        """Calculate metric."""
        return sklearn.metrics.mean_absolute_percentage_error(y_true, y_pred)


class MSE(NumericMetric):
    """Mean Square Error."""

    @classmethod
    def calc(cls, y_true: Y_TYPE, y_pred: Y_TYPE, **kwargs: Any) -> float:
        """Calculate metric."""
        return sklearn.metrics.mean_squared_error(y_true, y_pred, squared=True)


class RMSE(NumericMetric):
    """Root Mean Square Error."""

    @classmethod
    def calc(cls, y_true: Y_TYPE, y_pred: Y_TYPE, **kwargs: Any) -> float:
        """Calculate metric."""
        return sklearn.metrics.mean_squared_error(y_true,
                                                  y_pred,
                                                  squared=False)


class R2(NumericMetric):
    """R^2 Score."""

    @classmethod
    def calc(cls, y_true: Y_TYPE, y_pred: Y_TYPE, **kwargs: Any) -> float:
        """Calculate metric."""
        return sklearn.metrics.r2_score(y_true, y_pred)


class TargetType(Enum):
    """Target type."""

    REGRESSION = 'regression'


METRICS: dict[str, dict[str, Type[BaseMetric]]] = {
    TargetType.REGRESSION.value: {
        'mae': MAE,
        'mape': MAPE,
        'mse': MSE,
        'rmse': RMSE,
        'r2': R2,
    },
}
