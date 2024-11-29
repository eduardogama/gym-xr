

import os

import pandas as pd
import numpy as np


trace = np.array(
    pd.read_csv(
        f'{os.getcwd()}/trace.csv'
    ).values
).flatten()

print(trace[98])