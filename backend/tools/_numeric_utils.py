"""
Shared numeric-safety helpers.

`pandas.Series.std()`/`.mean()` return NaN for degenerate groups (e.g. a
single-transaction customer). The codebase previously guarded this with
`float(x.std() or 0)`, which does NOT work: NaN is truthy in Python (only
0, '', None, empty containers, etc. are falsy), so `nan or 0` evaluates to
`nan`, not `0`. That NaN then serializes into JSON as the literal token
`NaN`, which is invalid per the JSON spec and breaks any strict JSON
consumer (e.g. JS `JSON.parse`, many non-Python clients).

Use `safe_float(x, default)` instead of `x or default` anywhere a pandas
aggregate could be NaN.
"""
import math
from typing import Optional


def safe_float(value, default: float = 0.0) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(f) else f
