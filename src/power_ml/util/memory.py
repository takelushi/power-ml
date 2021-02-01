"""Memory."""

import os

import psutil

UNITS = ['B', 'KB', 'MB', 'GB', 'TB']


def get_process_memory(unit: str = 'B') -> int:
    """Get process memory size.

    Returns:
        int: Memory size.
    """
    n = UNITS.index(unit)
    v = psutil.Process(os.getpid()).memory_info().rss
    return v / (1024**n)
