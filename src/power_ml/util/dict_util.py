"""Dict utility."""

import hashlib
import json
from typing import Any, Union


def get(d: dict, keys: Union[str, list[str]], safe: bool = True) -> Any:
    """Get dict value by keys.

    Args:
        d (dict): Target dict.
        keys (List[str]): Keys.
        safe (bool, optional): Safe or not.

    Raises:
        KeyError: Not found key on safe=False.

    Returns:
        Any: Value.
    """
    if isinstance(keys, str):
        keys = [keys]
    result = d.copy()
    for k in keys:
        try:
            result = result[k]
        except TypeError as err:
            if safe:
                return None
            else:
                raise err
        except KeyError as err:
            if safe:
                return None
            else:
                raise err
    return result


def to_list(d: dict, keys_list: list[Union[str, list[str]]]) -> list:
    """Convert dict to list.

    Args:
        d (dict): Target dict.
        keys_list (List[Union[str, List[str]]]): Keys.
            e.g.
                [
                    'a',
                    ['b', 'b_child'],
                ]

    Returns:
        list: Result.
    """
    result = []

    for keys in keys_list:
        result.append(get(d, keys))

    return result


def to_json_value(v: Any) -> Any:
    """To JSON value.

    Args:
        v (Any): Any value.

    Returns:
        Any: JSON like value.
    """
    if isinstance(v, dict):
        res = {}
        for k, val in v.items():
            if not any(isinstance(k, t) for t in [str, int]):
                k = str(k)
            res[k] = to_json_value(val)
        return res
    if any(isinstance(v, t) for t in [str, int, float, bool]):
        return v
    elif any(isinstance(v, t) for t in [list, tuple, set]):
        return [to_json_value(val) for val in v]
    elif v is None:
        return v
    else:
        return str(v)


def to_str(d: dict, sort_keys: bool = False, indent: int = None) -> str:
    """To str.

    Args:
        d (dict): Target dict.
        sort_keys (bool, optional): Sort key or not.
        indent (int, optional): Indent size.

    Returns:
        str: String dict.
    """
    json_dict = to_json_value(d)
    return json.dumps(json_dict,
                      ensure_ascii=False,
                      sort_keys=sort_keys,
                      indent=indent)


def to_hash(d: dict) -> str:
    """To hash dict.

    Args:
        d (dict): Target dict.

    Returns:
        str: Hash string.
    """
    s = to_str(d, sort_keys=True)
    return hashlib.md5(s.encode('UTF-8')).hexdigest()
