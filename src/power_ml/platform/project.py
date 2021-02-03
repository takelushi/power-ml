"""Project."""

import logging
from typing import Type

import numpy as np

from power_ml.ai.metrics import BaseMetric
from power_ml.ai.model import Model
from power_ml.ai.predictor.base_predictor import BasePredictor
from power_ml.ai.validate.cv_generator import CVGenerator
from power_ml.data.store import BaseStore


class Project:
    """Project."""

    def __init__(self,
                 target_type: str,
                 metrics: list[Type[BaseMetric]],
                 store: BaseStore,
                 logger: logging.Logger = None) -> None:
        """Initialize object."""
        if logger is None:
            logger = logging.getLogger(self.__class__.__name__)
        self.logger = logger
        self.target_type = target_type
        self.metrics = metrics
        self.store = store
        self.models: dict[str, str] = {}
        self.info: dict = {}

    def get_model(self, predictor_hash: str) -> Model:
        """Get predictor.

        Args:
            predictor_hash (str): Predictor hash.

        Returns:
            BasePredictor: Predictor.
        """
        predictor_name = self.models[predictor_hash]
        # TODO:  Check type
        model: Model = self.store.load(predictor_name)
        return model

    def train(self, predictor_class: Type[BasePredictor], param: dict,
              x_name: str, y_name: str) -> str:
        """Train.

        TODO: Docstring
        """
        predictor = predictor_class(self.target_type, param)

        x = self.store.load(x_name)
        y = self.store.load(y_name)
        predictor_hash = predictor.hash_train(x, y)

        try:
            model = self.get_model(predictor_hash)
            name = self.models[predictor_hash]
        except KeyError:
            model = Model(predictor, self.metrics)
            model.train(x, y)
            name = self.store.save(model)

        self.models[predictor_hash] = name
        return predictor_hash

    def predict(self, predictor_hash: str, x_name: str) -> np.ndarray:
        """Predict.

        TODO: Docstring
        """
        x = self.store.load(x_name)
        model = self.get_model(predictor_hash)

        return model.predict(x)

    def evaluate(self, predictor_hash: str, x_name: str, y_name: str) -> dict:
        model = self.get_model(predictor_hash)
        x = self.store.load(x_name)
        y = self.store.load(y_name)
        return model.evaluate(x, y)[0]

    def train_cv(self, predictor_class: Type[BasePredictor], param: dict,
                 x_name: str, y_name: str, cv_type: str,
                 cv_n: int) -> list[str]:
        cv_generator = CVGenerator(cv_type, self.store)
        cv_names_li = []
        for res in cv_generator.generate(5, x_name, y_name):
            print(res)
            cv_names_li.append(res)
