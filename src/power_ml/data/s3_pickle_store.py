"""S3 pickle store."""

import logging
import os
from typing import Any

import boto3

from power_ml.data.local_pickle_store import LocalPickleStore
from power_ml.data.store import BaseStore

s3 = boto3.resource('s3')


def exist_key(bucket: str, key: str) -> bool:
    """Exist key or not.

    Args:
        bucket (str): S3 bucket name.
        key (str): Object key.

    Returns:
        bool: Exist or not.
    """
    try:
        s3.Object(bucket, key).get()
    except s3.meta.client.exceptions.NoSuchKey:
        return False
    return True


class S3PickleStore(BaseStore):
    """S3 store."""

    def __init__(
        self,
        bucket_name: str,
        tmp_path: str,
        cache: bool = True,
        region: str = None,
        exist: str = 'skip',
        logger: logging.Logger = None,
    ) -> None:
        """Initialize object."""
        if exist in ['skip', 'error']:
            self.exist = exist
        else:
            raise ValueError('Invalid exist parameter.')
        self.region = region
        self.bucket_name = bucket_name
        self.cache = cache
        kwargs = {}
        if self.region:
            kwargs['region_name'] = self.region
        super().__init__(logger=logger)
        self.pickle_store = LocalPickleStore(tmp_path, logger=self.logger)

    def init(self, **kwargs) -> None:
        """Init store."""
        super().init(**kwargs)

    def _save_data(self, data: Any, name: str, exist_ok: bool = True) -> None:
        """Save data.

        Args:
            data (Any): Data.
            name (str): Data name.
            exist_ok (bool, optional): Exist ok or not.
        """
        if self.exist == 'skip' and exist_key(self.bucket_name, name):
            return

        if self.exist == 'error' or exist_ok:
            if exist_key(self.bucket_name, name):
                raise FileExistsError('Key already exist. {}'.format(name))

        s3.Object(self.bucket_name, name).upload_file(data)

    def _load_data(self, name: str) -> Any:
        """Load data.

        Args:
            name (str): Name.

        Returns:
            Any: Data.
        """
        path = self.pickle_store._get_pickle_path(name)
        if self.cache and os.path.exists(path):
            data = self.pickle_store._load_data(name)
        else:
            s3.Object(self.bucket_name, name).download_file(path)
            data = self.pickle_store._load_data(name)
            self.pickle_store._remove_data(name)
        return data

    def _remove_data(self, name: str) -> None:
        """Remove data.

        Args:
            name (str): Name.
        """
        s3.Object(self.bucket_name, name).delete()

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
        name = self.pickle_store.save(obj,
                                      prefix=prefix,
                                      meta=meta,
                                      exist_ok=True)
        path = self.pickle_store._get_pickle_path(name)
        meta_name = self.pickle_store._get_meta_name(name)
        meta_path = self.pickle_store._get_pickle_path(meta_name)
        self._save_data(path, name, exist_ok=exist_ok)
        self._save_data(meta_path, meta_name, exist_ok=exist_ok)

        self.pickle_store._remove_data(name)
        self.pickle_store._remove_data(meta_name)
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
