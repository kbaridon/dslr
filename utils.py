import pandas as pd
from math import sqrt


def my_count(col) -> float:
    return len(col.dropna())


def my_mean(col) -> float:
    col = col.dropna()
    n = len(col)
    if n == 0:
        return float('nan')
    total = 0
    for value in col:
        total += value
    return (total / n)


def my_std(col) -> float:
    col = col.dropna()
    n = len(col)
    if n <= 1:
        return float('nan')
    mean = my_mean(col)
    total = 0
    for value in col:
        total += (mean - value) ** 2
    return sqrt(total / (n - 1))


def my_min(col) -> float:
    col = col.dropna()
    if len(col) == 0:
        return float('nan')
    mini = col.iloc[0]
    for value in col.iloc[1:]:
        if mini > value:
            mini = value
    return mini


def my_quantile(col, percent: float) -> float:
    col = col.dropna().sort_values()
    n = len(col)
    if n == 0:
        return float('nan')
    index = percent * (n - 1)
    lower = int(index)
    upper = lower + 1
    if upper >= n:
        return float(col.iloc[lower])
    fraction = index - lower
    return float(col.iloc[lower] + fraction * (col.iloc[upper] - col.iloc[lower]))


def my_max(col) -> float:
    col = col.dropna()
    if len(col) == 0:
        return float('nan')
    maxi = col.iloc[0]
    for value in col.iloc[1:]:
        if maxi < value:
            maxi = value
    return maxi
