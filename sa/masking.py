"""User-turn loss mask (pure Python, shared by all engines). Mirrors their model_runner.py:
spans from "<|im_start|>user" to the next "<|im_end|>" (inclusive), located in the shifted labels."""
from .prompts import TURN_END, USER_START


def _find(seq, pattern, start=0):
    n = len(pattern)
    for i in range(start, len(seq) - n + 1):
        if seq[i:i + n] == pattern:
            return i
    return -1


def user_mask(encode, labels, mask_user=True):
    """List of 1.0 (counts in the loss) / 0.0 (inside a user turn), one per label position."""
    labels = list(labels)
    mask = [1.0] * len(labels)
    if not mask_user:
        return mask
    start_ids, end_ids = encode(USER_START), encode(TURN_END)
    i = _find(labels, start_ids)
    while i != -1:
        j = _find(labels, end_ids, i + len(start_ids))
        stop = len(labels) if j == -1 else j + len(end_ids)
        for k in range(i, stop):
            mask[k] = 0.0
        i = _find(labels, start_ids, stop)
    return mask
