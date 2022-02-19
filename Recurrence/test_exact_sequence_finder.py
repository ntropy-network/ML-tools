import pytest
from exact_sequence_finder import (
    find_intervals_fixed_width,
    find_intervals,
)


@pytest.fixture
def sequences():
    seq2 = list(range(9, 34, 11))
    seq1 = list(range(11, 30, 5))
    seq3 = list(range(15, 34, 7))
    seq = sorted(seq1 + seq2 + seq3)
    return seq, seq1, seq2, seq3


def test_fixed_width_finder(sequences):
    seq, seq1, seq2, seq3 = sequences

    for arr, width in zip(
        [seq1, seq2, seq3],
        [5, 11, 7],
    ):
        candidates = find_intervals_fixed_width(
            arr=arr,
            width=width,
            return_values=True,
        )
        if all([v != arr for k, v in candidates.items()]):
            raise ValueError(f"Failed on: {arr}")


def test_finder(sequences):
    seq, seq1, seq2, seq3 = sequences

    candidates = find_intervals(seq, min_width=5, max_width=13, return_values=True)

    for x in [seq1, seq2, seq3]:
        failed = True
        for y in candidates:
            if len(x) <= len(y) and all([a == b for a, b in zip(x, y)]):
                failed = False
                break
        if failed:
            raise ValueError(f"Failed on: {x}")
