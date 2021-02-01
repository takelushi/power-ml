"""Power Model."""

from typing import Any, Iterable, Type

import numpy as np
import pandas as pd

from power_ml.ai.model import Model
from power_ml.ai.predictor.base_predictor import BasePredictor
from power_ml.ai.selection.selector import SELECTORS
from power_ml.ai.selection.split import split_index
from power_ml.data.store import BaseStore
from power_ml.stats.col_stats import get_col_stats
from power_ml.util.seed import set_seed, set_seed_random


class PowerModel:
    """Power Model."""

    def __init__(self,
                 target_type: str,
                 target: str,
                 store: BaseStore,
                 data: str,
                 seed: int = None) -> None:
        """Initialize object."""
        if seed is None:
            seed = set_seed_random()
        else:
            set_seed(seed)
        self.seed = seed
        self.target_type = target_type
        self.target = target
        self.store = store
        # TODO: Check data exist.

        if not isinstance(data, str):
            data = store.save(data)
        self._data: dict[str, Any] = {'master': data}

    def _register_trn_val(self,
                          data_type: str,
                          trn_idx: pd.Index,
                          val_idx: pd.Index = None) -> tuple[str, str]:
        """Register train and validation index."""
        if data_type not in ['base', 'validation']:
            raise ValueError(
                'Cannot register. data_type must be base or validation')

        master_data = self.store.load(self._data['master'])
        trn_idx_name = self.store.save(trn_idx)

        if val_idx is not None:
            val_idx_name = self.store.save(val_idx)
        else:
            val_idx = master_data.drop(trn_idx).index
            val_idx_name = self.store.save(val_idx)

        # Check index.
        _ = master_data.iloc[trn_idx]
        _ = master_data.iloc[val_idx]

        names = (trn_idx_name, val_idx_name)
        if data_type == 'base':
            self._data['base'] = names
        elif data_type == 'validation':
            self._data['validation'].append(names)
        return names

    def register_trn_val(self, trn_idx: pd.Index,
                         val_idx: pd.Index) -> tuple[str, str]:
        """Register data index."""
        if 'base' in self._data:
            raise ValueError('Already registered.')
        return self._register_trn_val('base', trn_idx, val_idx=val_idx)

    def register_validation(
        self, idx_list: Iterable[tuple[pd.Index,
                                       pd.Index]]) -> list[tuple[str, str]]:
        """Register validation data index."""
        if 'validation' in self._data:
            raise ValueError('Already registered.')

        self._data['validation'] = []
        names_li = []
        for idx in idx_list:
            names_li.append(
                self._register_trn_val('validation', idx[0], val_idx=idx[1]))

        return names_li

    def split_trn_val(self, train_ratio: float) -> tuple[str, str]:
        """Split train and validation.

        Args:
            train_ratio (float): Train ratio.

        Returns:
            str: Train index name.
            str: Validation index name.
        """
        set_seed(self.seed)
        if 'base' in self._data:
            raise ValueError('Already registered.')

        master_data = self.store.load(self._data['master'])
        trn_idx, val_idx = split_index(master_data, train_ratio)

        return self.register_trn_val(trn_idx, val_idx)

    def create_validation_index(self,
                                selector_name: str,
                                param: dict = None) -> list[tuple[str, str]]:
        """Create validation index."""
        set_seed(self.seed)
        if 'validation' in self._data:
            raise ValueError('Already registered.')

        selector_class: type = SELECTORS[selector_name]
        if param is None:
            param = {}
        master_data = self.store.load(self._data['master'])

        selector = selector_class(**param)

        return self.register_validation(selector.split(master_data))

    def calc_column_stats(self) -> pd.DataFrame:
        """Calculate column stats."""
        master_data = self.store.load(self._data['master'])
        self.col_stats = get_col_stats(master_data)
        return self.col_stats

    def train(
        self,
        predictor_class: Type[BasePredictor],
        idx_name: str,
        param: dict = None,
    ) -> str:
        """Train."""
        set_seed(self.seed)
        predictor = predictor_class(self.target_type, param=param)
        # If model exist use exist model and not train.
        model = Model(predictor)
        data_idx = self.store.load(idx_name)
        data = self.store.load(self._data['master']).iloc[data_idx]
        x = data.drop(columns=[self.target])
        y = data[self.target]
        model.train(x, y)
        model_name = self.store.save(model)
        return model_name

    def predict(self, model_name: str, idx_name: str) -> str:
        """Predict."""
        set_seed(self.seed)
        model: Model = self.store.load(model_name)
        data_idx = self.store.load(idx_name)
        data = self.store.load(self._data['master']).iloc[data_idx]
        x = data.drop(columns=[self.target])

        y_pred = model.predict(x)
        y_pred_name = self.store.save(y_pred)
        return y_pred_name

    def validate(self, model_name: str, idx_name: str) -> tuple[dict, str]:
        """Validate."""
        set_seed(self.seed)
        model: Model = self.store.load(model_name)
        data_idx = self.store.load(idx_name)
        data = self.store.load(self._data['master']).iloc[data_idx]
        x = data.drop(columns=[self.target])
        y = data[self.target]
        score, y_pred = model.validate(x, y)
        y_pred_name = self.store.save(y_pred)
        return score, y_pred_name

    def train_validate(
        self,
        predictor_class: Type[BasePredictor],
        trn_idx_name: str,
        tst_idx_name: str,
        train_param: dict = None,
    ) -> tuple[str, dict, str]:
        """Train and validate."""
        model_name = self.train(predictor_class,
                                trn_idx_name,
                                param=train_param)
        score, y_pred_name = self.validate(model_name, tst_idx_name)
        return model_name, score, y_pred_name

    def validate_each(
        self,
        predictor_class: Type[BasePredictor],
        idx_li: list[tuple[str, str]],
        train_param: dict = None,
    ) -> tuple[float, list]:
        results = []
        scores: dict[str, list[float]] = {}
        for idx in idx_li:
            result = self.train_validate(predictor_class,
                                         idx[0],
                                         idx[1],
                                         train_param=train_param)
            results.append(result)
            for metric, score in result[1].items():
                scores[metric] = scores.get(metric, [])
                scores[metric].append(score)
        score_mean: dict[str, float] = {}
        for metric in scores.keys():
            score_mean[metric] = np.mean(scores[metric])

        return score_mean, results
