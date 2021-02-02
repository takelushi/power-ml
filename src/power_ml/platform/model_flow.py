"""Model flow."""

from typing import Any, Iterable, Optional, Type, TypedDict

import numpy as np
import pandas as pd

from power_ml.ai.metrics import BaseMetric
from power_ml.ai.model import Model
from power_ml.ai.predictor.base_predictor import BasePredictor
from power_ml.ai.selection.selector import BaseSelector, SELECTORS
from power_ml.ai.selection.split import split_index
from power_ml.platform.catalog import Catalog
from power_ml.stats.col_stats import get_col_stats
from power_ml.util.seed import set_seed, set_seed_random


class Partition(TypedDict):
    """Partition."""

    names: tuple[str, str]
    type: str
    param: dict


class CVPartition(TypedDict):
    """CV Partition."""

    cv_names: list[tuple[str, str]]
    type: str
    param: dict


class ModelFlow:
    """Model flow."""

    def __init__(self,
                 target_type: str,
                 predictor_class: Type[BasePredictor],
                 target: str,
                 catalog: Catalog,
                 data_id: str,
                 metrics: list[Type[BaseMetric]] = None,
                 train_param: dict = None,
                 seed: int = None) -> None:
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
        self.metrics = metrics
        self.train_param = {} if train_param is None else train_param

        self.score: dict = {}
        self.perm: dict = {}

        self.data_id = data_id
        self.model_id: Optional[str] = None
        self.partition: Optional[Partition] = None
        self.cv_partition: Optional[CVPartition] = None

        self.cv_model_ids: Optional[list[str]] = None

    def _get_model(self) -> Model:
        predictor = self.predictor_class(self.target_type, self.train_param)
        return Model(predictor, metrics=self.metrics)

    def create_partition(self, partition_type: str,
                         **kwargs) -> tuple[str, str]:
        set_seed(self.seed)
        if self.partition is not None:
            raise RuntimeError('Partition already created.')
        data = self.catalog.load_table(self.data_id)

        if partition_type == 'random':
            trn_ratio = kwargs['trn_ratio']
            val_ratio = kwargs.get('val_ratio', None)

            if trn_ratio <= 0 or 1 <= trn_ratio:
                raise ValueError('The ratio must be 0 < v < 1.')
            if val_ratio is not None:
                if val_ratio <= 0 or 1 <= val_ratio:
                    raise ValueError('The ratio must be 0 < v < 1.')
                if trn_ratio + val_ratio > 1:
                    raise ValueError('The total ratio must be less equal 1.')

            trn_size = round(data.shape[0] * trn_ratio)
            trn_idx = data.sample(trn_size).index
            if val_ratio is None:
                val_idx = data.drop(trn_idx).index
            else:
                val_size = round(data.shape[0] * val_ratio)
                val_idx = data.drop(trn_idx).sample(val_size).index
            trn_idx_id = self.catalog.save_index(self.data_id, trn_idx)
            val_idx_id = self.catalog.save_index(self.data_id, val_idx)
            names = trn_idx_id, val_idx_id
        elif partition_type == 'manual':
            trn_idx_id = kwargs['trn_idx_id']
            val_idx_id = kwargs.get('val_idx_id', None)
            names = self._register_partition(trn_idx_id, val_idx_id=val_idx_id)
        else:
            raise ValueError('Unknown partition_type.')

        self.partition = Partition(names=names,
                                   type=partition_type,
                                   param=kwargs)

        return trn_idx_id, val_idx_id

    def _register_partition(self,
                            trn_idx_id: str,
                            val_idx_id: str = None) -> tuple[str, str]:
        data = self.catalog.load_table(self.data_id)
        trn_idx = self.catalog.load_index(trn_idx_id)

        if val_idx_id is not None:
            val_idx = self.catalog.load_index(val_idx_id)
        else:
            val_idx = data.drop(trn_idx).index
            val_idx_id = self.catalog.save_index(self.data_id, val_idx)

        # Check index.
        _ = data.iloc[trn_idx]
        _ = data.iloc[val_idx]

        return trn_idx_id, val_idx_id

    def _train(self, trn_idx_id: str) -> str:
        set_seed(self.seed)
        data = self.catalog.load_table(self.data_id)
        data_idx = self.catalog.load_index(trn_idx_id)
        sample = data.iloc[data_idx]
        x = sample.drop(columns=[self.target])
        y = sample[self.target]
        model = self._get_model()
        model_id = model.hash_model(x, y)
        try:
            model = self.catalog.load_model(model_id)
        except IndexError:
            model.train(x, y)
            new_model_id = self.catalog.save_model(model)
            assert new_model_id == model_id

        return model_id

    def train(self) -> str:
        if self.partition is None:
            raise RuntimeError('Need to create partition.')

        if self.model_id is not None:
            raise RuntimeError('Already trained.')

        trn_idx_id, _ = self.partition['names']

        self.model_id = self._train(trn_idx_id)

        return self.model_id

    def _validate(self, model_id: str, val_idx_id: str) -> dict:
        set_seed(self.seed)
        data = self.catalog.load_table(self.data_id)
        model = self.catalog.load_model(model_id)
        data_idx = self.catalog.load_index(val_idx_id)
        sample = data.iloc[data_idx]
        x = sample.drop(columns=[self.target])
        y = sample[self.target]

        score, y_pred = model.validate(x, y)
        # TODO: Save y_pred?
        return score

    def validate(self, mode='both') -> dict:
        if self.partition is None:
            raise RuntimeError('Need to create partition.')
        if self.model_id is None:
            raise RuntimeError('Need to train model.')

        # if  is not None:
        #     raise RuntimeError('Already validated.')
        model_id = self.model_id
        names = self.partition['names']
        result = {}

        def _f(use_train: bool):
            idx = names[0] if use_train else names[1]
            key = 'train' if use_train else 'validation'
            self.score[key] = self._validate(model_id, idx)
            result[key] = self.score[key]

        if mode == 'train':
            _f(True)
        elif mode == 'validation':
            _f(False)
        elif mode == 'both':
            _f(True)
            _f(False)
        else:
            raise ValueError('Invalid mode.')

        return result

    def train_validate(self) -> tuple[str, dict]:
        model_id = self.train()
        score = self.validate()
        return model_id, score

    def create_cv_partition(self, cv_type: str,
                            **kwargs) -> list[tuple[str, str]]:
        if self.cv_partition is not None:
            raise RuntimeError('CV partition already created.')
        set_seed(self.seed)
        data = self.catalog.load_table(self.data_id)

        selector_class = SELECTORS.get(cv_type.lower(), None)
        if selector_class is not None:
            selector: BaseSelector = selector_class(**kwargs)
            cv_names = []
            for idx in selector.split(data):
                trn_idx = pd.Index(idx[0])
                val_idx = pd.Index(idx[1])
                idx_ids = (
                    self.catalog.save_index(self.data_id, trn_idx),
                    self.catalog.save_index(self.data_id, val_idx),
                )
                cv_names.append(idx_ids)
        elif cv_type == 'manual':
            cv_names = kwargs['partition_idx_ids']
            # self._register_partition(idx_ids[0], val_idx_id=idx_ids[1])
        else:
            raise ValueError('Unknown cv_type.')

        self.cv_partition = CVPartition(cv_names=cv_names,
                                        type=cv_type,
                                        param=kwargs)

        return cv_names

    def train_cv(self) -> list[str]:
        if self.cv_partition is None:
            raise RuntimeError('Need to create cv partition.')

        if self.cv_model_ids is not None:
            raise RuntimeError('Already trained.')

        model_ids = []
        for names in self.cv_partition['cv_names']:
            trn_idx_id, _ = names
            model_id = self._train(trn_idx_id)
            model_ids.append(model_id)
        self.cv_model_ids = model_ids

        return self.cv_model_ids

    def _validate_cv(self, use_train: bool = False) -> tuple[dict, list[dict]]:
        if self.cv_partition is None:
            raise RuntimeError('Need to create cv partition.')
        if self.cv_model_ids is None:
            raise RuntimeError('Need to train cv model.')

        # if  is not None:
        #     raise RuntimeError('Already validated.')
        scores_li = []
        cv_model_ids = self.cv_model_ids
        cv_names = self.cv_partition['cv_names']

        tmp_scores: dict[str, list[float]] = {}
        for model_id, names in zip(cv_model_ids, cv_names):
            idx_id = names[0] if use_train else names[1]
            scores = self._validate(model_id, idx_id)
            for metric, score in scores.items():
                tmp_scores[metric] = tmp_scores.get(metric, [])
                tmp_scores[metric].append(score)
            scores_li.append(scores)

        score_mean: dict[str, float] = {}
        for metric in tmp_scores.keys():
            score_mean[metric] = np.mean(tmp_scores[metric])
        return score_mean, scores_li

    def validate_cv(self, mode: str = 'both') -> dict:
        result = {}

        def _f(use_train: bool):
            key = 'train' if use_train else 'validation'
            cv_key = 'cv_{}'.format(key)
            each_key = 'cv_{}_each'.format(key)
            score_mean, each_score = self._validate_cv(use_train=use_train)
            self.score[cv_key] = score_mean
            self.score[each_key] = each_score
            result[cv_key] = self.score[cv_key]
            result[each_key] = self.score[each_key]

        if mode == 'train':
            _f(True)
        elif mode == 'validation':
            _f(False)
        elif mode == 'both':
            _f(True)
            _f(False)
        else:
            raise ValueError('Invalid mode.')

        return result

    def cross_validation(self) -> tuple[list[str], dict]:
        cv_model_ids = self.train_cv()
        score = self.validate_cv()
        return cv_model_ids, score

    def _calc_perm(self, model_id: str, idx_id: str,
                   **kwargs) -> dict[str, pd.DataFrame]:
        set_seed(self.seed)
        model = self.catalog.load_model(model_id)
        data_idx = self.catalog.load_index(idx_id)
        data = self.catalog.load_table(self.data_id).iloc[data_idx]
        x = data.drop(columns=[self.target])
        y = data[self.target]
        result = model.calc_perm(x, y, **kwargs)
        # TODO: Save perm
        return result

    def calc_perm(self, mode: str = 'both', **kwargs) -> dict:
        if self.partition is None:
            raise RuntimeError('Need to create partition.')
        if self.model_id is None:
            raise RuntimeError('Need to train model.')

        names = self.partition['names']
        result = {}

        model_id = self.model_id

        def _f(use_train: bool):
            idx = names[0] if use_train else names[1]
            key = 'train' if use_train else 'validation'
            self.perm[key] = self._calc_perm(model_id, idx, **kwargs)
            result[key] = self.perm[key]

        if mode == 'train':
            _f(True)
        elif mode == 'validation':
            _f(False)
        elif mode == 'both':
            _f(True)
            _f(False)
        else:
            raise ValueError('Invalid mode.')

        return result

    def calc_cv_perm(self, mode: str = 'both', **kwargs) -> dict:
        if self.cv_partition is None:
            raise RuntimeError('Need to create cv partition.')
        if self.cv_model_ids is None:
            raise RuntimeError('Need to train cv model.')

        result = {}

        cv_model_ids = self.cv_model_ids
        cv_partition = self.cv_partition['cv_names']

        def _f_mean_perm(perms_li: list[dict]) -> dict:
            mean_perm: dict[str, pd.DataFrame] = {}
            metrics = set()
            cols = ['Weight', 'Score']
            for perms in perms_li:
                for metric, perm in perms.items():
                    metrics.add(metric)
                    if mean_perm.get(metric) is None:
                        mean_perm[metric] = perm[['Column'] + cols].copy()
                    else:
                        mean_perm[metric][cols] += perm[cols]

            for metric in list(metrics):
                df = mean_perm[metric]
                df[cols] /= len(perms_li)
                df['Top'] = df['Weight'].rank(ascending=False).astype(int)
                f = lambda x: 'no' if x == 1 else 'better' if x > 1 else 'worse'  # noqa: E731,E501
                df['Type'] = df['Weight'].apply(f)
                mean_perm[metric] = df.sort_values(by=['Weight'],
                                                   ascending=False)

            return mean_perm

        def _f(use_train: bool):
            key = 'train' if use_train else 'validation'
            cv_key = 'cv_{}'.format(key)
            each_key = 'cv_{}_each'.format(key)

            perms = []
            for model_id, names in zip(cv_model_ids, cv_partition):
                idx = names[0] if use_train else names[1]
                perm = self._calc_perm(model_id, idx, **kwargs)
                perms.append(perm)
            self.perm[cv_key] = _f_mean_perm(perms)
            self.perm[each_key] = perms
            result[cv_key] = self.perm[cv_key]
            result[each_key] = self.perm[each_key]

        if mode == 'train':
            _f(True)
        elif mode == 'validation':
            _f(False)
        elif mode == 'both':
            _f(True)
            _f(False)
        else:
            raise ValueError('Invalid mode.')

        return result
