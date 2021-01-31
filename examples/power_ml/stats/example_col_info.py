"""Example of column info."""

import pandas as pd

from power_ml.stats.col_info import get_col_info

url = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv'  # noqa: E501

df = pd.read_csv(url)
col_info = get_col_info(df)

print(col_info)
