"""Catalog."""

from typing import Any

import joblib
import pandas as pd
from tinydb import Query, TinyDB

from power_ml.ai.model import Model
from power_ml.data.store import BaseStore


class Catalog:
    """Catalog."""

    def __init__(self, db_path: str, store: BaseStore) -> None:
        """Initialize object."""
        self.db = TinyDB(db_path)
        self.store = store

    def _save(self,
              table_name: str,
              row: dict,
              data: Any,
              overwrite: bool = True) -> None:
        table = self.db.table(table_name)
        q = Query()
        res = table.search(q.id == row['id'])
        if not overwrite and len(res) == 0:
            data_name = self.store.save(data)
            row['store_name'] = data_name
            table.insert(row)

    def _load(self, table_name: str, _id: str) -> None:
        table = self.db.table(table_name)
        q = Query()
        store_name = table.search(q.id == _id)[0]['store_name']
        return self.store.load(store_name)

    def save_table(self, data: Any) -> str:
        if isinstance(data, pd.Index):
            raise TypeError('The index should be save with save_index().')
        data_id = joblib.hash(data)
        self._save('table', {'id': data_id}, data, overwrite=False)
        return data_id

    def load_table(self, data_id: str) -> Any:
        return self._load('table', data_id)

    def save_index(self, data_id: str, idx: pd.Index) -> str:
        if not isinstance(idx, pd.Index):
            raise TypeError('Not index. type: {}'.format(type(idx)))
        index_id = joblib.hash(idx)
        row = {'id': index_id, 'data_id': data_id}
        self._save('index', row, idx, overwrite=False)
        return index_id

    def load_index(self, index_id: str) -> pd.Index:
        return self._load('index', index_id)

    def save_model(self, model: Model) -> str:
        # TODO: Is instance BaseModel?
        model_id = model.predictor_hash
        self._save('model', {'id': model_id}, model)
        return model_id

    def load_model(self, model_id) -> Model:
        return self._load('model', model_id)
