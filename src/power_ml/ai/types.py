"""Types."""

from typing import Union

import numpy as np
import pandas as pd

X_TYPE = Union[list[list], np.ndarray, pd.DataFrame]
Y_TYPE = Union[list, np.ndarray, pd.Series]
