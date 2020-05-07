"""Demonstration of resampling"""

import pandas as pd

from Experiment.visualizer import visualize
from egsmote import EGSmote
from collections import Counter
import numpy as np


# Read CSV file
df = pd.read_csv('../Experiment/data/NSLKDD-mini.csv')

X = np.asarray(df.iloc[:, :-1].values)
y = np.asarray(df.iloc[:, -1].values)

print('Original dataset shape %s' % (Counter(y)))
# Original dataset shape Counter({{1: 28416, 0: 150}})

egsmote = EGSmote(sampling_rate=0.25)
X_res, y_res = egsmote.fit_resample(X, y)

print('Resampled dataset shape %s' % (Counter(y_res)))
# Resampled dataset shape Counter({{0: 28416, 1: 9472}})

visualize(X,y,X_res,y_res)
# Visualize the resampled data - check output folder