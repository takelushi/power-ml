"""CV generator."""

from typing import Callable, Dict, Iterable, Tuple, Union

import sklearn.model_selection

from power_ml.data.store import BaseStore

CV_SPLITTER: Dict[str, Callable] = {
    k.lower(): v for k, v in {
        'KFold': sklearn.model_selection.KFold,
        'StratifiedKFold': sklearn.model_selection.StratifiedKFold,
    }.items()
}


class CVGenerator:
    """Cross validation data generator."""

    def __init__(self, splitter: Union[str, Callable], store: BaseStore):
        """Initialize object."""
        if isinstance(splitter, str):
            splitter = CV_SPLITTER[splitter.lower()]
        self.splitter = splitter
        self.store = store

    def generate(self, n_splits: int, x_name: str,
                 y_name: str) -> Iterable[Tuple[str, str, str, str]]:
        """Generate CV data.

        Args:
            n_splits (int): Splits.
            x_name (str): X data name.
            y_name (str): Y data name.

        Yields:
            str: X train data name.
            str: Y train data name.
            str: X test data name.
            str: Y test data name.
        """
        x = self.store.load(x_name)
        y = self.store.load(y_name)
        for idx in self.splitter(n_splits=n_splits).split(x, y=y):
            x_trn_name = self.store.save(x.iloc[idx[0]])
            y_trn_name = self.store.save(y.iloc[idx[0]])
            x_tst_name = self.store.save(x.iloc[idx[1]])
            y_tst_name = self.store.save(y.iloc[idx[1]])
            yield x_trn_name, y_trn_name, x_tst_name, y_tst_name
