"""Power Model."""

from typing import Any, Type

from power_ml.ai.predictor.base_predictor import BasePredictor
from power_ml.ai.selection.selector import SELECTORS
from power_ml.ai.selection.split import split_index
from power_ml.platform.catalog import Catalog
from power_ml.platform.model_flow import ModelFlow
from power_ml.util.seed import set_seed, set_seed_random


class PowerModel:
    """Power Model."""

    def __init__(self,
                 target_type: str,
                 predictor_class: Type[BasePredictor],
                 target: str,
                 catalog: Catalog,
                 data: Any,
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
        self.model_flow = ModelFlow(self.target_type,
                                    predictor_class,
                                    target,
                                    catalog,
                                    data,
                                    train_param=train_param,
                                    seed=seed)
        # TODO: Check data exist.

        # if isinstance(data, str):
        #    data = self.catalog.store.load(data)

        # data_id = self.catalog.save_table(data)
        # self._data: dict[str, Any] = {'master': data_id}

    def run_flow(self, name: str, **kwargs) -> Any:
        func = getattr(self.model_flow, name)
        return func(**kwargs)

    def split_trn_val(self, train_ratio: float) -> tuple[str, str]:
        """Split train and validation.

        Args:
            train_ratio (float): Train ratio.

        Returns:
            str: Train index name.
            str: Validation index name.
        """
        set_seed(self.seed)
        if 'base' in self.model_flow._data:
            raise ValueError('Already registered.')

        master_data = self.catalog.load_table(self.model_flow._data['master'])
        trn_idx, val_idx = split_index(master_data, train_ratio)

        return self.model_flow.register_trn_val(trn_idx, val_idx)

    def create_validation_index(self,
                                selector_name: str,
                                param: dict = None) -> list[tuple[str, str]]:
        """Create validation index."""
        set_seed(self.seed)
        if 'validation' in self.model_flow._data:
            raise ValueError('Already registered.')

        selector_class: type = SELECTORS[selector_name]
        if param is None:
            param = {}
        master_data = self.catalog.load_table(self.model_flow._data['master'])

        selector = selector_class(**param)

        return self.model_flow.register_validation(selector.split(master_data))
