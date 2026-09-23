#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def canonical_pair(a, b):
    """Canonical orientation used consistently across subjective and objective data."""
    a, b = str(a), str(b)
    return (a, b) if a <= b else (b, a)


def wilson_ci(k, n, alpha=0.05):
    if n <= 0:
        return np.nan, np.nan
    z = norm.ppf(1 - alpha / 2)
    p = k / n
    den = 1 + z*z/n
    center = (p + z*z/(2*n)) / den
    half = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return max(0.0, center-half), min(1.0, center+half)


def parse_bool(v):
    if pd.isna(v):
        return np.nan
    if isinstance(v, (bool, np.bool_)):
        return int(v)
    if isinstance(v, (int, np.integer, float, np.floating)) and not pd.isna(v):
        return int(float(v) != 0.0)
    s = str(v).strip().lower()
    if s in {"true", "1", "yes", "y", "correct"}:
        return 1
    if s in {"false", "0", "no", "n", "incorrect"}:
        return 0
    raise ValueError(f"Could not interpret boolean value {v!r}")


def percentile_ci(values, alpha=0.05):
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan
    return tuple(np.quantile(x, [alpha/2, 1-alpha/2]))
