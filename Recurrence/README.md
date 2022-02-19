# Recurrence

## Exact methods

The exact_... files contain some exact algorithms. These are fast and scalable, but may not be completely useful, as real sequences in the wild often have fuzzy intervals.

We present a simple utility to find the maximal subsequences of integers with regular intervals. Note, that these are maximal intervals, meaning that if you spliced together some recurring sequences to form your main sequence, this may find spurious patterns. But, that's expected, as the merge operation is non-reversible (we are just picking solutions that optimize for length of sequence).

For complete details, please see the accompanying blog post at <LINK>


