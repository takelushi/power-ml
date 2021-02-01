"""Local pickle store."""

import datetime
import hashlib
import logging
import os
import pickle
import time
from typing import Any, Optional
import uuid

from power_ml.data.store import BaseStore


class LocalPickleStore(BaseStore):
    """Local pickle store."""

    def __init__(self,
                 root: str,
                 logger: logging.Logger = None,
                 **kwargs) -> None:
        """Initialize object."""
        self.root = root
        self.root = os.path.abspath(self.root)
        super().__init__(logger=logger, root=root)

    def init(self, **kwargs) -> None:
        """Init store."""
        os.makedirs(self.root, exist_ok=True)

    def _get_pickle_path(self, name: str) -> str:
        """Get pickle path.

        Args:
            name (str): Name.

        Returns:
            str: Pickle path.
        """
        return os.path.join(self.root, '{}.pickle'.format(name))

    def _save_data(self, data: Any, name: str, exist_ok: bool = True) -> None:
        """Save data.

        Args:
            data (Any): Data.
            name (str): Data name.
            exist_ok (bool, optional): Exist ok or not.
        """
        path = self._get_pickle_path(name)

        if not exist_ok and os.path.exists(path):
            raise FileExistsError('Path: "{}"'.format(path))
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def _load_data(self, name: str) -> Any:
        """Load data.

        Args:
            name (str): Name.

        Returns:
            Any: Data.
        """
        path = self._get_pickle_path(name)
        last_exc: Optional[Exception] = None
        wait_time = 1.0
        for _ in range(5):
            try:
                with open(path, 'rb') as f:
                    return pickle.load(f)
            except Exception as exc:  # noqa: B902
                last_exc = exc
            self.logger.warning('An {} has occurred. Retry...'.format(
                last_exc.__class__.__name__))
            time.sleep(wait_time)
            wait_time *= 2
        if last_exc:
            raise last_exc

    def _remove_data(self, name: str) -> None:
        """Remove data.

        Args:
            name (str): Name.
        """
        path = self._get_pickle_path(name)
        os.remove(path)

    def _hash_file(self, name: str) -> str:
        """Hash file.

        Args:
            name (str): Target name.

        Returns:
            str: Hash string.
        """
        path = self._get_pickle_path(name)
        return hashlib.sha256(open(path, 'rb').read()).hexdigest()

    def _rename(self,
                src_name: str,
                dst_name: str,
                exist_ok: bool = True) -> None:
        """Rename file.

        Args:
            src_name (str): Source name.
            dst_name (str): Destination name.
        """
        src_path = self._get_pickle_path(src_name)
        dst_path = self._get_pickle_path(dst_name)
        if not exist_ok and os.path.exists(dst_path):
            raise FileExistsError('Path: "{}"'.format(dst_path))
        os.rename(src_path, dst_path)

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
        if meta is None:
            meta = {}
        meta['type'] = obj.__class__
        meta['str'] = str(obj)[:100]

        tmp_name = 'tmp_{}'.format(uuid.uuid4())
        self._save_data(obj, tmp_name, exist_ok=exist_ok)

        name = self._hash_file(tmp_name)
        if prefix:
            if not prefix.endswith('_'):
                prefix += '_'
            name = prefix + name

        try:
            self._rename(tmp_name, name, exist_ok=exist_ok)
            meta['created_at'] = datetime.datetime.now()
            self.save_meta(meta, name, exist_ok=exist_ok)
        except FileExistsError as exc:
            self._remove_data(tmp_name)
            raise exc

        self.logger.debug('Saved: "{}", {}'.format(name, meta))
        return name

    def load_with_meta(  # type: ignore
            self,
            name: str,
            expected_type: type = None,
            validate: bool = True) -> tuple[Any, dict]:
        """Load object with meta.

        Args:
            name (str): Name.
            expected_type (type, optional): Expected type.
            validate (bool, optional): Validate with meta flag.

        Returns:
            Any: Object.
            dict: Meta.
        """
        obj = self._load_data(name)

        if expected_type:
            assert isinstance(obj, expected_type)

        meta = self.load_meta(name)
        if validate:
            assert isinstance(obj, meta['type'])
            assert meta['str'] == str(obj)[:100]

        self.logger.debug('Loaded: "{}"'.format(name))
        return obj, meta
