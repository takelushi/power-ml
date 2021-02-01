"""Power Model."""

from typing import Any, Iterable, Type

import numpy as np
import pandas as pd

from power_ml.ai.model import Model
from power_ml.ai.predictor.base_predictor import BasePredictor
from power_ml.ai.selection.selector import SELECTORS
from power_ml.ai.selection.split import split_index
from power_ml.data.store import BaseStore
from power_ml.platform.catalog import Catalog
from power_ml.stats.col_stats import get_col_stats
from power_ml.util.seed import set_seed, set_seed_random


class PowerModel:
    """Power Model."""

    def __init__(self,
                 target_type: str,
                 target: str,
                 store: BaseStore,
                 data: Any,
                 seed: int = None) -> None:
        """Initialize object."""
        if seed is None:
            seed = set_seed_random()
        else:
            set_seed(seed)
        self.seed = seed
        self.target_type = target_type
        self.target = target
        self.catalog = Catalog('./tmp/catalog.json', store)
        # TODO: Check data exist.

        if isinstance(data, str):
            data = store.load(data)

        data_id = self.catalog.save_table(data)
        self._data: dict[str, Any] = {'master': data_id}

    def _register_trn_val(self,
                          data_type: str,
                          trn_idx: pd.Index,
                          val_idx: pd.Index = None) -> tuple[str, str]:
        """Register train and validation index."""
        if data_type not in ['base', 'validation']:
            raise ValueError(
                'Cannot register. data_type must be base or validation')
        data_id = self._data['master']
        master_data = self.catalog.load_table(data_id)
        trn_idx_id = self.catalog.save_index(data_id, trn_idx)

        if val_idx is not None:
            val_idx_id = self.catalog.save_index(data_id, val_idx)
        else:
            val_idx = master_data.drop(trn_idx).index
            val_idx_id = self.catalog.save_index(data_id, val_idx)

        # Check index.
        _ = master_data.iloc[trn_idx]
        _ = master_data.iloc[val_idx]

        names = (trn_idx_id, val_idx_id)
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
        ids_li = []
        for idx in idx_list:
            ids_li.append(
                self._register_trn_val('validation', idx[0], val_idx=idx[1]))

        return ids_li

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

        master_data = self.catalog.load_table(self._data['master'])
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
        master_data = self.catalog.load_table(self._data['master'])

        selector = selector_class(**param)

        return self.register_validation(selector.split(master_data))

    def calc_column_stats(self) -> pd.DataFrame:
        """Calculate column stats."""
        master_data = self.catalog.load_table(self._data['master'])
        self.col_stats = get_col_stats(master_data)
        return self.col_stats

    def train(
        self,
        predictor_class: Type[BasePredictor],
        idx_id: str,
        param: dict = None,
    ) -> str:
        """Train."""
        set_seed(self.seed)
        predictor = predictor_class(self.target_type, param=param)
        # If model exist use exist model and not train.
        model = Model(predictor)
        data_idx = self.catalog.load_index(idx_id)
        data = self.catalog.load_table(self._data['master']).iloc[data_idx]
        x = data.drop(columns=[self.target])
        y = data[self.target]
        model_id = model._predictor.hash_train(x, y)
        try:
            model = self.catalog.load_model(model_id)
        except IndexError:
            model.train(x, y)
            self.catalog.save_model(model)
        return model_id

    def predict(self, model_id: str, idx_id: str) -> str:
        """Predict."""
        set_seed(self.seed)
        model = self.catalog.load_model(model_id)
        data_id = self._data['master']
        data_idx = self.catalog.load_index(idx_id)
        data = self.catalog.load_table(data_id).iloc[data_idx]
        x = data.drop(columns=[self.target])

        y_pred = model.predict(x)
        y_pred_id = self.catalog.save_table(y_pred)
        return y_pred_id

    def validate(self, model_id: str, idx_id: str) -> tuple[dict, str]:
        """Validate."""
        set_seed(self.seed)
        model = self.catalog.load_model(model_id)
        data_id = self._data['master']
        data_idx = self.catalog.load_index(idx_id)
        data = self.catalog.load_table(data_id).iloc[data_idx]
        x = data.drop(columns=[self.target])
        y = data[self.target]
        score, y_pred = model.validate(x, y)
        y_pred_id = self.catalog.save_table(y_pred)
        return score, y_pred_id

    def train_validate(
        self,
        predictor_class: Type[BasePredictor],
        trn_idx_id: str,
        tst_idx_id: str,
        train_param: dict = None,
    ) -> tuple[str, dict, str]:
        """Train and validate."""
        model_id = self.train(predictor_class, trn_idx_id, param=train_param)
        score, y_pred_name = self.validate(model_id, tst_idx_id)
        return model_id, score, y_pred_name

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

    def calc_perm(self, model_id: str, idx_id: str,
                  n: int) -> dict[str, pd.DataFrame]:
        set_seed(self.seed)
        model = self.catalog.load_model(model_id)
        data_id = self._data['master']
        data_idx = self.catalog.load_index(idx_id)
        data = self.catalog.load_table(data_id).iloc[data_idx]
        x = data.drop(columns=[self.target])
        y = data[self.target]
        result = model.calc_perm(x, y, n=n)
        self.catalog.save_model(model)
        return result
