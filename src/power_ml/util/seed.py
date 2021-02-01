"""Seed."""

import os
import random


def set_seed(value: int) -> None:
    """Set seed.

    Args:
        value (int): Seed value.
    """
    random.seed(value)
    os.environ['PYTHONHASHSEED'] = str(value)

    try:
        import numpy as np
        np.random.seed(value)
    except ModuleNotFoundError:
        pass

    try:
        import tensorflow as tf
        tf.random.set_seed(value)
    except ModuleNotFoundError:
        pass


def set_seed_random(min_value: int = 0, max_value: int = 99999999) -> int:
    """Set seed randomly.

    Args:
        min_value (int, optional): Minimum value. Defaults to 0.
        max_value (int, optional): Maximum value. Defaults to 99999999.

    Returns:
        int: Seed value.
    """
    value = random.randint(min_value, max_value)

    set_seed(value)

    return value
