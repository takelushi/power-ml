"""Selector."""

from abc import ABC, abstractmethod
from typing import Iterable, Union

import pandas as pd
import sklearn.model_selection

from power_ml.ai.types import X_TYPE, Y_TYPE
from power_ml.util.seed import set_seed


class BaseSelector(ABC):
    """Base selector."""

    @abstractmethod
    def split(self,
              x: X_TYPE,
              y: Y_TYPE = None) -> Iterable[tuple[pd.Index, pd.Index]]:
        """Split."""
        raise NotImplementedError()


SKLEARN_SELECTION: dict[str, type] = {
    k.lower(): v for k, v in {
        'KFold': sklearn.model_selection.KFold,
        'StratifiedKFold': sklearn.model_selection.StratifiedKFold,
    }.items()
}


class SklearnSelector(BaseSelector):
    """Sklearn Selector."""

    def __init__(self, selection: Union[str, type], **param: dict) -> None:
        """Initialize object."""
        if isinstance(selection, str):
            self.selection = SKLEARN_SELECTION[selection.lower()]
        else:
            self.selection = selection

        self._splitter = self.selection(**param)

    def split(self,
              x: X_TYPE,
              y: Y_TYPE = None) -> Iterable[tuple[pd.Index, pd.Index]]:
        """Split."""
        for idx in self._splitter.split(x, y=y):
            yield idx[0], idx[1]


class SeedSampleSelector(BaseSelector):
    """Seed sample selector."""

    def __init__(self,
                 seeds: list[int],
                 trn_ratio: float,
                 val_ratio: float = None) -> None:
        """Initialize object."""
        self.seeds = seeds
        if trn_ratio <= 0 or trn_ratio >= 1:
            raise ValueError(
                'ratio must be 0 < x < 1. ratio: {}'.format(trn_ratio))
        if val_ratio:
            if val_ratio <= 0 or val_ratio >= 1:
                raise ValueError(
                    'ratio must be 0 < x < 1. ratio: {}'.format(val_ratio))

            if (total_ratio := trn_ratio + val_ratio) > 1:
                raise ValueError(
                    'trn_ratio + val_ratio <= 1. {}'.format(total_ratio))

        self.trn_ratio = trn_ratio
        self.val_ratio = val_ratio

    def split(self,
              x: X_TYPE,
              y: Y_TYPE = None) -> Iterable[tuple[pd.Index, pd.Index]]:
        """Split."""
        for seed in self.seeds:
            set_seed(seed)
            trn_size = round(x.shape[0] * self.trn_ratio)
            trn_idx = x.sample(trn_size).index
            if self.val_ratio:
                val_size = round(x.shape[0] * self.val_ratio)
                val_idx = x.drop(trn_idx).sample(val_size).index
            else:
                val_idx = x.drop(trn_idx).index

            yield trn_idx, val_idx


SELECTORS: dict[str, type] = {
    'sklearn': SklearnSelector,
    'seed': SeedSampleSelector,
}
