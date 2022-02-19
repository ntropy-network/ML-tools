from typing import List, Dict


def find_intervals_fixed_width(
    arr: List[int], width: int, return_values: bool = False
) -> Dict[int, List[int]]:
    """
    Args:
      arr: a list of integers, strictly increasing order.
      width: the specified interval size to look for patterns.
      return_values: whether to return lists of the repeating sequences by value
                     or wether to return lists of indicies into the original array.
    Returns:
      a dict of lists of integers, containing either values, or the indices in the
      original input array :arr:. the keys of the dictionary indicate the starting
      value of the sequence.
    """
    seqs = {}
    nexts = {}
    for i, x in enumerate(arr):
        nexts[x + width] = x
        seqs[x] = [x if return_values else i]
        if x in nexts:
            seqs[nexts[x]].append(x if return_values else i)
            nexts[x + width] = nexts[x]
    return seqs


def find_intervals(
    arr: List[int], max_width: int, min_width: int = 1, return_values: bool = False
) -> List[List[int]]:
    """
    Args:
      arr: a list of integers, strictly increasing order.
      max_width: max allowed separation between integers in a sequence.
      return_values: whether to return lists of the repeating sequences by value
                     or wether to return lists of indicies into the original array.
    Returns:
      a list of lists of integers, containing either values, or the indices in the
      original input array :arr:.
    """
    assert 0 < min_width < max_width, min_width
    assert 0 < max_width, max_width
    if len(arr) < 1:
        return []
    max_width = min(max_width, arr[-1] - arr[0])
    res = []
    for width in range(min_width, max_width + 1):
        intervals = find_intervals_fixed_width(
            arr=arr, width=width, return_values=return_values
        )
        for k, v in intervals.items():
            if len(v) > 1:
                res.append(v)
    res.sort(key=len, reverse=True)
    return res
