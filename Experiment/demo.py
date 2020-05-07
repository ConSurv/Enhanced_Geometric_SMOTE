import pandas as pd
from egsmote import EGSmote
from collections import Counter

# Read CSV file
df = pd.read_csv('../Experiment/data/NSLKDD-mini.csv')

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

print('Original dataset shape %s' % (Counter(y)))
# Original dataset shape Counter({{1: 28416, 0: 150}})

egsmote = EGSmote(sampling_rate=0.25)
X_res, y_res = egsmote.fit_resample(X, y)

print('Resampled dataset shape %s' % (Counter(y_res)))
# Resampled dataset shape Counter({{0: 28416, 1: 9472}})
