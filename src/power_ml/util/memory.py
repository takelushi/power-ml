"""Memory."""

import sys


def show_memory_usage() -> dict[str, int]:
    """Show memory usage."""
    print('{}{: >25}{}{: >10}{}'.format('|', 'Variable Name', '|', 'Memory',
                                        '|'))
    print(' ------------------------------------ ')
    result = {}
    for name in dir():
        if not name.startswith('_'):
            size = sys.getsizeof(eval(name))
            result[name] = size
            print('{}{: >25}{}{: >10}{}'.format('|', name, '|', size, '|'))
    return result
