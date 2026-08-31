from __future__ import annotations

import numpy as np


def apply_temperature(
    probabilities: np.ndarray,
    temperature: float,
) -> np.ndarray:
    """Apply scalar temperature scaling to probability vectors."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    p = np.asarray(probabilities, dtype=float)

    if p.ndim != 2:
        raise ValueError("probabilities must be a 2-D array")

    if not np.isfinite(p).all():
        raise ValueError("probabilities contain non-finite values")

    p = np.clip(p, 1e-12, 1.0)

    z = np.log(p) / float(temperature)
    z = z - np.max(z, axis=1, keepdims=True)

    exp_z = np.exp(z)
    calibrated = exp_z / np.sum(
        exp_z,
        axis=1,
        keepdims=True,
    )

    return calibrated
