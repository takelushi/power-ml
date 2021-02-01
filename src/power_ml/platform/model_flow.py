"""Model flow."""

from typing import Any, Iterable, Type

import numpy as np
import pandas as pd

from power_ml.ai.model import Model
from power_ml.ai.predictor.base_predictor import BasePredictor
from power_ml.platform.catalog import Catalog
from power_ml.stats.col_stats import get_col_stats
from power_ml.util.seed import set_seed, set_seed_random


class ModelFlow:
    """Model flow."""

    def __init__(self,
                 target_type: str,
                 predictor_class: Type[BasePredictor],
                 target: str,
                 catalog: Catalog,
                 data: Any,
                 train_param: dict = None,
                 seed: int = None,
                 n_perms: int = 1) -> None:
        """Initialize object."""
        if seed is None:
            seed = set_seed_random()
        else:
            set_seed(seed)
        self.seed = seed
        self.target_type = target_type
        self.target = target
        self.catalog = catalog
        self.predictor_class = predictor_class
        self.train_param = {} if train_param is None else train_param
        self.n_perms = n_perms

        # TODO: Check data exist.
        if isinstance(data, str):
            data = self.catalog.store.load(data)

        data_id = self.catalog.save_table(data)

        self._data: dict[str, Any] = {'master': data_id}
        self._models: dict[str, Any] = {}
        self._perms: dict[str, Any] = {'base': {}, 'validation': {}}

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

    def calc_column_stats(self) -> pd.DataFrame:
        """Calculate column stats."""
        master_data = self.catalog.load_table(self._data['master'])
        self.col_stats = get_col_stats(master_data)
        return self.col_stats

    def _train(
        self,
        idx_id: str,
    ) -> str:
        """Train."""
        set_seed(self.seed)
        predictor = self.predictor_class(self.target_type,
                                         param=self.train_param)
        model = Model(predictor)
        data_idx = self.catalog.load_index(idx_id)
        data = self.catalog.load_table(self._data['master']).iloc[data_idx]
        x = data.drop(columns=[self.target])
        y = data[self.target]
        model_id = model.hash_model(x, y)
        try:
            model = self.catalog.load_model(model_id)
        except IndexError:
            model.train(x, y)
            self.catalog.save_model(model)
        self._models['base'] = model_id
        return model_id

    def train(self) -> str:
        """Train."""
        return self._train(self._data['base'][0])

    def _predict(self, model_id: str, idx_id: str) -> str:
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

    def predict(self) -> str:
        """Predict."""
        model_id = self._models['base']
        idx_id = self._data['base'][1]
        return self._predict(model_id, idx_id)

    def _validate(self, model_id: str, idx_id: str) -> tuple[dict, str]:
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

    def validate(self) -> tuple[dict, str]:
        """Validate."""
        model_id = self._models['base']
        idx_id = self._data['base'][1]
        return self._validate(model_id, idx_id)

    def train_validate(
        self,
        trn_idx_id: str,
        tst_idx_id: str,
    ) -> tuple[str, dict, str]:
        """Train and validate."""
        model_id = self._train(trn_idx_id)
        score, y_pred_name = self._validate(model_id, tst_idx_id)
        return model_id, score, y_pred_name

    def validate_each(self) -> tuple[float, list]:
        results = []
        scores: dict[str, list[float]] = {}
        for idx in self._data['validation']:
            result = self.train_validate(idx[0], idx[1])
            results.append(result)
            for metric, score in result[1].items():
                scores[metric] = scores.get(metric, [])
                scores[metric].append(score)
        score_mean: dict[str, float] = {}
        for metric in scores.keys():
            score_mean[metric] = np.mean(scores[metric])

        return score_mean, results

    def _calc_perm(self, model_id: str,
                   idx_id: str) -> dict[str, pd.DataFrame]:
        set_seed(self.seed)
        model = self.catalog.load_model(model_id)
        data_id = self._data['master']
        data_idx = self.catalog.load_index(idx_id)
        data = self.catalog.load_table(data_id).iloc[data_idx]
        x = data.drop(columns=[self.target])
        y = data[self.target]
        result = model.calc_perm(x, y, n=self.n_perms)
        self.catalog.save_model(model)
        return result

    def calc_perm(self, mode: str) -> dict[str, pd.DataFrame]:
        if mode not in ['train', 'validation']:
            raise ValueError('mode must be train or validation.')
        idx_pair = self._data['base']
        idx = idx_pair[0] if mode == 'train' else idx_pair[1]
        perm = self._calc_perm(self._models['base'], idx)
        self._perms['base'][mode] = perm
        return perm

    def calc_perm_each(self, mode: str) -> dict[str, pd.DataFrame]:
        if mode not in ['train', 'validation']:
            raise ValueError('mode must be train or validation.')
        idx_pair_li = self._data['validation']
        if mode == 'train':
            idx_li = [idx_pair[0] for idx_pair in idx_pair_li]
        else:
            idx_li = [idx_pair[1] for idx_pair in idx_pair_li]

        perms = {}
        for idx in idx_li:
            perm = self._calc_perm(self._models['base'], idx)

            for metric, p in perm.items():
                if metric not in perms:
                    perms[metric] = {}

                    for row in p.itertuples():
                        perms[metric][row[1]] = row[3]
                else:
                    for row in p.itertuples():
                        perms[metric][row[1]] += row[3]
        self._perms['validation'][mode] = perms
        return perms
