"""Version."""

import sys
from typing import Iterator

DEFAULT_VERSION_TARGETS = {
    'Power ML': 'power_ml',
    'IPython': 'IPython',
    'Numpy': 'numpy',
    'Pandas': 'pandas',
    'Matplotlib': 'matplotlib',
    'Scikit-learn': 'sklearn',
    'Tensorflow': 'tensorflow',
    'ipywidgets': 'ipywidgets',
    'tqdm': 'tqdm',
}


def get_versions(
        targets: dict[str, str] = None) -> Iterator[tuple[str, bool, str]]:
    """Get module versions.

    Args:
        targets (dict[str, str], optional): Target modules.

    Yields:
        str: Target name.
        bool: Exists or not.
        str: Version.
    """
    if targets is None:
        targets = DEFAULT_VERSION_TARGETS

    for name, module_name in targets.items():

        try:
            module = __import__(module_name)
        except ModuleNotFoundError:
            exists = False
            version = '-'
        else:
            exists = True

            try:
                version = module.__version__
            except AttributeError:
                version = 'Unknown'

        yield name, exists, str(version)


def show_versions(targets: dict[str, str] = None,
                  py: bool = True,
                  to_df: bool = True):
    """Show module versions.

    Args:
        targets (dict[str, str], optional): Target modules.
        py (bool, optional): Print Python version or not.
        to_df (bool, optional): Show with DataFrame or not.
    """
    if py:
        print(f'Python:\n{sys.version}'.replace('\n', '\n    '))

    headers = ['Name', 'Available', 'Version']
    col_size_li = [len(header) for header in headers]
    rows: list[list[str]] = []

    for name, exists, version in get_versions(targets=targets):
        row: list[str] = [name, 'Yes' if exists else 'No', version]
        for i in range(len(row)):
            col_size_li[i] = max(col_size_li[i], len(row[i]))
        rows.append(row)

    if to_df:
        try:
            import pandas as pd
            df = pd.DataFrame(rows, columns=headers)

            try:
                import IPython.display
                IPython.display.display(df)
            except ImportError:
                print(df)
            return
        except ImportError:
            pass

    rows = [headers, ['-' * size for size in col_size_li]] + rows

    lines = []
    for name, available, version in rows:
        fmt = '| '
        fmt += ' | '.join([f'{{:<{size}}}' for size in col_size_li])
        fmt += ' |'
        lines.append(fmt.format(name, available, version))

    print('\n'.join(lines))
