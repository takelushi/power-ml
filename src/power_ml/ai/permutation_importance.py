"""Permutation Importance."""

from abc import ABC
from typing import Iterable, Type

import combu
import numpy as np
import pandas as pd

from power_ml.ai.metrics import NumericMetric
from power_ml.ai.predictor.base_predictor import BasePredictor
from power_ml.ai.types import X_TYPE, Y_TYPE


class PermutationImportance(ABC):
    """Permutation Importance."""

    def __init__(
        self,
        predictor: BasePredictor,
        metric: Type[NumericMetric],
        x: pd.DataFrame,
        y: Y_TYPE,
        n: int = 1,
        n_jobs: int = 1,
    ) -> None:
        """Initialize object.

        Args:
            predictor (BasePredictor): Perdictor.
            metric (Type[NumericMetric]): Metric.
            x (pd.DataFrame): X.
            y (Y_TYPE): Y.
            n (int, optional): Shuffle and evaluate count.
        """
        self.predictor = predictor
        self.metric = metric
        self.x = x
        self.y = y
        self.score = self.evaluate(self.x, self.y)
        self.n = n

    def shuffle_and_evaluate(self, col: str) -> float:
        """Shuffle and evaluate.

        Args:
            col (str): Target column.

        Returns:
            float: Score.
        """
        total = 0.0
        for _ in range(self.n):
            x_shuffle = self.get_shuffled_x(col)
            total += self.evaluate(x_shuffle, self.y)
        return total / self.n

    def iter_perm(self, n_jobs: int = 1) -> Iterable[tuple[str, float, float]]:
        """Iterate Permutation Importance.

        Args:
            n_jobs (int, optional): Parallel num. Defaults to 1.

        Yields:
            str: Column name.
            float: Weight. (Shuffuled score / real score)
            float: Score.
        """
        iter_comb = combu.execute(self.shuffle_and_evaluate,
                                  {'col': list(self.x.columns)},
                                  n_jobs=n_jobs)
        for result in iter_comb:
            score = result[0]
            col = result[1]['col']
            yield col, score / self.score, score

    def calc(self, n_jobs: int = 1) -> pd.DataFrame:
        """Calculate  Permutation Importance.

        Args:
            n_jobs (int, optional): Parallel num. Defaults to 1.

        Returns:
            pd.DataFrame: Result.
        """
        perms = list(self.iter_perm(n_jobs=n_jobs))
        df = pd.DataFrame(perms, columns=['Column', 'Weight', 'Score'])
        df = df.sort_values(by=['Weight'], ascending=False)
        df['Top'] = df['Weight'].rank(ascending=False).astype(int)
        f = lambda x: 'no' if x == 1 else 'better' if x > 1 else 'worse'  # noqa: E731,E501
        df['Type'] = df['Weight'].apply(f)

        return df

    def evaluate(self, x: X_TYPE, y_true: Y_TYPE) -> float:
        """Evaluate.

        Args:
            x (X_TYPE): X.
            y_true (Y_TYPE): Y true.

        Returns:
            np.ndarray: Score.
        """
        y_pred = self.predictor.predict(x)
        return self.metric(y_true, y_pred)  # type: ignore

    def get_shuffled_x(self, col: str) -> pd.DataFrame:
        """Get shuffled X.

        Args:
            col (str): Column.

        Returns:
            pd.DataFrame: Result.
        """
        x_shuffle = self.x.copy()
        x_shuffle[col] = np.random.permutation(x_shuffle[col].values)
        return x_shuffle
