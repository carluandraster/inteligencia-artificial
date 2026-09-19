"""Tipado mínimo para skfuzzy.cluster."""

from typing import Any

import numpy as np


def cmeans(
    data: Any,
    c: int,
    m: float,
    error: float,
    maxiter: int,
    init: Any = ...,
    seed: int | None = ...,
) -> tuple[
    np.ndarray,
    np.ndarray,
    float,
    float,
    np.ndarray,
    float,
    int,
]: ...
