"""Formatting."""

import math

UNITS: dict[str, str] = {
    'nano': 'ns',  # nanosecond
    'micro': 'µs',  # microsecond
    'ms': 'ms',  # millisecond
    'sec': 's',  # second
    'min': 'min',  # minute
    'hr': 'h',  # hour
    'd': 'd',  # day
}


def format_timespan(timespan):
    """Format time span.

    Args:
        timespan (float): Time span.
    Returns:
        str: Formatted string.
    """
    if timespan >= 60.0:
        parts = [(UNITS['d'], 60 * 60 * 24), (UNITS['hr'], 60 * 60),
                 (UNITS['min'], 60), (UNITS['sec'], 1)]
        time = []
        leftover = timespan
        for suffix, length in parts:
            value = int(leftover / length)
            if value > 0:
                leftover = leftover % length
                time.append(u'%s %s' % (str(value), suffix))
            if leftover < 1:
                break
        return ' '.join(time)

    units = [UNITS[k] for k in ['sec', 'ms', 'micro', 'nano']]
    scaling = [1, 1e3, 1e6, 1e9]

    if timespan > 0.0:
        order = min(-int(math.floor(math.log10(timespan)) // 3), 3)
    else:
        order = 3
    return '%.*g %s' % (3, timespan * scaling[order], units[order])
