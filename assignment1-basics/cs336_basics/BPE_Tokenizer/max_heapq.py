"""
This max-heap implementation utilizes the API in Python's built-in heapq module, 
which natively provides a min-heap
(see: https://stackoverflow.com/questions/33024215/built-in-max-heap-api-in-python)
In Python 3.11, the _heapq C module already includes C implementations of `heappop_max` and `heapify_max`,
namely `_HEAPQ__HEAPPOP_MAX_METHODEF` and `_HEAPQ__HEAPIFY_MAX_METHODDEF`.

Note that Python 3.14 already provides a native max-heap implementation in heapq, 
(see: https://github.com/python/cpython/blob/3.14/Modules/_heapqmodule.c, specifically the `heapq_methods`)
but this code is intended for compatibility with earlier versions of Python.

Usage:

heap = []                # creates an empty heap
heappush_max(heap, item) # pushes a new item on the heap
item = heappop_max(heap) # pops the largest item from the heap
item = heap[0]           # largest item on the heap without popping it
heapify_max(x)           # transforms list into a max-heap, in-place, in linear time
"""

import heapq

def heappush_max(heap, item):
    """Maxheap version of a heappush.
    Directly copied from https://github.com/python/cpython/blob/3.14/Lib/heapq.py.
    This function is not available in Python versions prior to 3.14,
    but interestingly, the internal API `siftdown_max` is available in Python 3.11."""
    heap.append(item)
    # Original code: _siftdown_max(heap, 0, len(heap)-1)
    heapq._siftdown_max(heap, 0, len(heap)-1)

def heappop_max(heap):
    """Maxheap version of a heappop.
    Directly use the API in https://github.com/python/cpython/blob/3.11/Lib/heapq.py."""
    return heapq._heappop_max(heap)

def heapify_max(heap):
    """Transform list into a maxheap, in-place, in O(len(x)) time.
    Directly use the API in https://github.com/python/cpython/blob/3.11/Lib/heapq.py."""
    heapq._heapify_max(heap)