import numpy as np
import time
import matplotlib.pyplot as plt


# Runtime:
#           average: O(n log(n))
#           worst case: O(n^2) (if partitioned elements are far from the median)
#
# Memory: O(log(n))
#
# QuickSort is a Divide and Conquer algorithm.
# It picks an element as pivot and partitions the given array around the picked pivot.

def partition(arr: list, idx_low: int, idx_high: int) -> int:
    # Chose the last element as the pivot
    pivot = arr[idx_high]
    # Initiate the pivot index
    i = idx_low - 1
    for j in range(idx_low, idx_high):
        # If the current element is lass than or equal to the pivot
        if arr[j] <= pivot:
            # Move the pivot index forward
            i = i + 1
            # Swap current element with element at the pivot index
            arr[i], arr[j] = arr[j], arr[i]
    # Move the pivot element to the correct pivot position
    i = i + 1
    arr[i], arr[idx_high] = arr[idx_high], arr[i]
    # Return the pivot index
    return i


def quick_sort(arr: list, idx_low: int, idx_high: int) -> None:
    if len(arr) <= 1:
        return
        # Ensure indices are not negatives and in correct order
    if idx_low < 0 or idx_low >= idx_high:
        return
    # Partition array and get the pivot index
    p = partition(arr, idx_low, idx_high)
    # Sort the two partitions
    quick_sort(arr, idx_low, p - 1)  # Left side of pivot
    quick_sort(arr, p + 1, idx_high)  # Right side of pivot


def is_sorted(arr: list) -> bool:
    for i in range(len(arr) - 1):
        if arr[i] > arr[i + 1]:
            return False
    return True


def test_quick_sort():
    array_tests = [
        [],
        [1],
        [1, 1],
        [1, 2],
        [2, 1],
        list(range(10)),
        list(reversed(range(10))),
    ]

    for arr in array_tests:
        n = len(arr)
        quick_sort(arr, 0, n - 1)
        assert len(arr) == n
        assert is_sorted(arr)


def time_complexity():
    step=100
    max_size = 1000
    times = []
    rng =  range(0, max_size + step, step)
    for n in rng:
        arr = np.random.randint(n, size=n)
        s = time.time()
        quick_sort(arr, 0, n-1)
        e = time.time()
        times.append(e - s)

    plt.plot(list(rng), times)


# EXAMPLE:
# arr[] = {5, 70, 40, 80, 20, 60}
# Indexes: 0   1   2   3   4   5
#
# low = 0, high =  5, pivot = arr[high] = 60
# Initialize i = -1
#
# Traverse elements from j = low to high-1
# j = 0 : Since arr[j] <= pivot, increment i and swap arr[i]=5 with arr[j]=5 (No change as i and j are same)
# --> i = 0
# --> arr = {5, 70, 40, 80, 20, 60}
#
# j = 1 : Since arr[j] > pivot, do nothing
#
# j = 2 : Since arr[j] <= pivot, increment i and swap arr[i]=70 with arr[j]=40
# --> i = 1
# --> arr = {5, 40, 70, 80, 20, 60}
#
# j = 3 : Since arr[j] > pivot, do nothing
#
# j = 4 : Since arr[j] <= pivot, increment i and swap arr[i]=70 with arr[j]=20
# --> i = 2
# --> arr = {5, 40, 20, 70, 80, 60}
#
# We come out of loop and place pivot at correct position by swapping arr[i+1]=80 and arr[high]=60
#  --> arr = {5, 40, 20, 60, 70, 80}
#
# Now all elements smaller than 60 are before it and all elements greater than 60 are after it.
