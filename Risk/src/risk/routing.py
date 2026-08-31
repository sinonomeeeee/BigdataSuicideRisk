from __future__ import annotations

from typing import Any

import numpy as np


def token_count_with_specials(
    texts: list[str],
    tokenizer: Any,
    batch_size: int = 256,
) -> np.ndarray:
    """
    Count full, untruncated tokenizer length including special tokens.
    """
    lengths: list[int] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]

        encoded = tokenizer(
            batch,
            add_special_tokens=True,
            truncation=False,
            padding=False,
        )

        lengths.extend(
            len(input_ids)
            for input_ids in encoded["input_ids"]
        )

    return np.asarray(lengths, dtype=np.int64)


def long_context_mask(
    lengths: np.ndarray,
    threshold: int = 384,
) -> np.ndarray:
    lengths = np.asarray(lengths, dtype=np.int64)
    return lengths > int(threshold)
