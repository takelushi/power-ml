"""Data split."""

from typing import Union

import pandas as pd


def split_index(data: pd.DataFrame,
                a_ratio: float,
                b_ratio: float = None) -> tuple[pd.Index, pd.Index]:
    """Split index.

    Args:
        data (pd.DataFrame): Data.
        ratio (float): The A data ratio.

    Returns:
        pd.Index: The A data index.
    """
    if a_ratio <= 0 or a_ratio >= 1:
        raise ValueError('ratio must be 0 < x < 1. ratio: {}'.format(a_ratio))
    if b_ratio is not None:
        if b_ratio <= 0 or 1 <= b_ratio:
            raise ValueError(
                'ratio must be 0 < x < 1. ratio: {}'.format(b_ratio))
        if (total_ratio := a_ratio + b_ratio) > 1:
            raise ValueError(
                'The total ratio must be less equal 1. {}'.format(total_ratio))

    a_size = round(data.shape[0] * a_ratio)
    a_idx = data.sample(a_size).index

    if b_ratio is None:
        b_idx = data.drop(a_idx).index
    else:
        b_size = round(data.shape[0] * b_ratio)
        b_idx = data.drop(a_idx).sample(b_size).index
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
