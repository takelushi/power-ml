"""Numeric utility."""

import decimal


def round_float(v: float, digits: int) -> float:
    """Round float.

    Args:
        v (float): Target.
        digits (int): Digits.

    Returns:
        float: Result.
    """
    digits_str = '.' + ('0' * digits)
    return float(
        decimal.Decimal(str(v)).quantize(decimal.Decimal(digits_str),
                                         rounding=decimal.ROUND_HALF_UP))
