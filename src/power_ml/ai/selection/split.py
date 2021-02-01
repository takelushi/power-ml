"""Data split."""

from typing import Union

import pandas as pd


def split_index(data: pd.DataFrame, ratio: float) -> tuple[pd.Index, pd.Index]:
    """Split index.

    Args:
        data (pd.DataFrame): Data.
        ratio (float): The A data ratio.

    Returns:
        pd.Index: The A data index.
    """
    if ratio <= 0 or ratio >= 1:
        raise ValueError('ratio must be 0 < x < 1. ratio: {}'.format(ratio))
    sample_size = round(data.shape[0] * ratio)
    a_idx = data.sample(sample_size).index
    b_idx = data.drop(a_idx).index
    return a_idx, b_idx


def split_with_index(
    data: Union[pd.DataFrame, pd.Series],
    index: pd.Index,
) -> Union[tuple[pd.DataFrame, pd.DataFrame], tuple[pd.Series, pd.Series]]:
    """Split with index.

    Args:
        data (Union[pd.DataFrame, pd.Series]): Data.
        index (pd.Index): The A index.

    Returns:
        Union[
            tuple[pd.DataFrame, pd.DataFrame],
            tuple[pd.Series, pd.Series]
        ]: The A and The B data.
    """
    a = data.iloc[index]
    b = data.drop(a.index)
    return a, b
