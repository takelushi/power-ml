"""Catalog."""

from power_ml.data.store import BaseStore


class Catalog:
    """Catalog."""

    def __init__(self, db_path: str, store: BaseStore) -> None:
        """Initialize object."""
        self.store = store

    def load_table(self, name: str) -> pd.DataFrame:
