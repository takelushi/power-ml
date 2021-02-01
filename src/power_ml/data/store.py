"""Store."""

from abc import ABC, abstractmethod
import logging
from typing import Any


class BaseStore(ABC):
    """Store data."""

    def __init__(self, logger: logging.Logger = None, **kwargs) -> None:
        """Initialize object."""
        if logger is None:
            logger = logging.getLogger(self.__class__.__name__)
        self.logger = logger

        self.init(**kwargs)

    @abstractmethod
    def init(self, **kwargs) -> None:
        """Init store."""
        pass

    def _get_meta_name(self, obj_name: str) -> str:
        """Get meta name.

        Args:
            obj_name (str): Object name.

        Returns:
            str: Meta name.
        """
        return '{}.meta'.format(obj_name)

    @abstractmethod
    def _save_data(self, data: Any, name: str, exist_ok: bool = True) -> None:
        """Save data.

        Args:
            data (Any): Data.
            name (str): Data name.
            exist_ok (bool, optional): Exist ok or not.
        """
        raise NotImplementedError()

    @abstractmethod
    def _load_data(self, name: str) -> Any:
        """Load data.

        Args:
            name (str): Name.

        Returns:
            Any: Data.
        """
        raise NotImplementedError()

    @abstractmethod
    def _remove_data(self, name: str) -> None:
        """Remove data.

        Args:
            name (str): Name.
        """
        raise NotImplementedError()

    def save_meta(self, meta: dict, name: str, exist_ok: bool = True) -> str:
        """Save meta.

        Args:
            meta (dict): Meta.
            name (str): Name.
            exist_ok (bool, optional): Exist ok or not.

        Returns:
            str: Meta name.
        """
        meta_name = self._get_meta_name(name)
        self._save_data(meta, meta_name, exist_ok=exist_ok)

        return meta_name

    def load_meta(self, name: str) -> dict:
        """Load meta.

        Args:
            data_name (str): Name.

        Returns:
            dict: Meta.
        """
        meta_name = self._get_meta_name(name)
        return self._load_data(meta_name)

    @abstractmethod
    def save(self,
             obj: Any,
             prefix: str = None,
             meta: dict = None,
             exist_ok: bool = True) -> str:
        """Save object.

        Args:
            obj (Any): Object.
            prefix (str, optional): Prefix of name.
            meta (dict, optional): Additional meta.
            exist_ok (bool, optional): Exist ok or not.

        Returns:
            str: Name.
        """
        raise NotImplementedError()

    @abstractmethod
    def load_with_meta(self, name: str, **kwargs) -> tuple[Any, dict]:
        """Load object with meta.

        Args:
            name (str): Name.

        Returns:
            Any: Object.
            dict: Meta.
        """
        raise NotImplementedError()

    def load(self, name: str, **kwargs) -> Any:
        """Load object.

        Args:
            name (str): Name.
            expected_type (type, optional): Expected type.
            validate (bool, optional): Validate with meta flag.

        Returns:
            Any: Object.
        """
        obj, _ = self.load_with_meta(name, **kwargs)

        return obj

    def delete(self, name: str) -> None:
        """Delete data file.

        Args:
            name (str): Name.
        """
        meta_name = self._get_meta_name(name)
        self._remove_data(meta_name)
        self._remove_data(name)
