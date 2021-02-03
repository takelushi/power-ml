"""Analyze Column Stats."""

from typing import Iterable

import numpy as np
import pandas as pd

from power_ml.util.numeric import round_float


def calc_col_stats(df: pd.DataFrame) -> Iterable[tuple[str, dict]]:
    """Calculate column infos.

    Args:
        df (pd.DataFrame): Data.

    Yields:
        str: Column name.
        dict: Column info.
    """
    for col in df.columns:
        col_info = {}
        s = df[col]
        col_info['dtype'] = str(s.dtype)
        is_numeric = \
            ('int' in col_info['dtype'] or 'float' in col_info['dtype'])
        col_info['numeric'] = int(is_numeric)  # type: ignore
        col_info['nunique'] = s.nunique()
        col_info['null'] = s.isnull().sum()
        col_info['null_%'] = col_info['null'] / df.shape[0]
        if col_info['numeric'] == 1:
            col_info['min'] = s.min()
            for p in [5, 25, 50, 75, 95]:
                col_info['{}p'.format(p)] = s.quantile(p * 0.01)
            col_info['max'] = s.max()
            col_info['mean'] = s.mean()
            col_info['std'] = s.std()
        else:
            pass

        uniques = s.value_counts().reset_index().values
        for i in range(1, 6):
            if len(uniques) < i:
                break
            name = 'u_{:02d}'.format(i)
            col_info['{}_v'.format(name)] = uniques[i - 1][0]
            col_info['{}_%'.format(name)] = \
                uniques[i - 1][1] / df.shape[0]

        yield col, col_info


def pretty_col_stats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace(np.nan, '-', regex=True)

    def _f(v):
        if isinstance(v, float) or isinstance(v, int):
            return round_float(v * 100, 2)
        else:
            return v

    for col in df.columns:
        if '%' in col:
            df[col] = df[col].apply(_f)

    return df


def get_col_stats(df: pd.DataFrame, pretty: bool = True) -> pd.DataFrame:
    """Get column info.

    Args:
        df (pd.DataFrame): Data.

    Returns:
        pd.DataFrame: Column info.
    """
    col_infos = {col: col_info for col, col_info in calc_col_stats(df)}

    result = pd.DataFrame([{'column': k, **v} for k, v in col_infos.items()])
    if pretty:
        result = pretty_col_stats(result)
    return result
