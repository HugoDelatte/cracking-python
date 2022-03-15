import sys
import numpy as np
import time
import pandas as pd

pd.options.plotting.backend = "plotly"


# Runtime:
#           average: O(n log(n))
#           worst case: O(n^2) (by using lomuto algo if partitioned elements are far from the median)
#
# Memory: O(log(n))
#
# QuickSort is a Divide and Conquer algorithm.
# It picks an element as pivot and partitions the given array around the picked pivot.

# ======================================================================================================================
#                                             Median of three
# ======================================================================================================================
def median_of_three(arr: list, low: int, high: int) -> None:
    # Estimate of the median
    mid = (high + low) // 2
    if arr[mid] < arr[low]:
        arr[mid], arr[low] = arr[low], arr[mid]
    if arr[high] < arr[low]:
        arr[high], arr[low] = arr[low], arr[high]
    # Swap the median estimate in the last position
    if arr[mid] < arr[high]:
        arr[mid], arr[high] = arr[high], arr[mid]


# ======================================================================================================================
#                                          Lomuto partition scheme
# ======================================================================================================================
def partition_lomuto(arr: list, low: int, high: int, use_last: bool = False) -> int:
    if not use_last:
        # Gives a better estimate of the optimal pivot (the true median) than selecting any single element
        median_of_three(arr, low, high)
    # Chose the last element as the pivot
    pivot = arr[high]
    # Initiate the pivot index
    i = low - 1
    for j in range(low, high):
        # If the current element is lass than or equal to the pivot
        if arr[j] <= pivot:
            # Move the pivot index forward
            i = i + 1
            # Swap current element with element at the pivot index
            arr[i], arr[j] = arr[j], arr[i]
    # Move the pivot element to the correct pivot position
    i = i + 1
    arr[i], arr[high] = arr[high], arr[i]
    # Return the pivot index
    return i


def quick_sort_lomuto(arr: list, low: int, high: int, use_last: bool) -> None:
    if len(arr) <= 1:
        return
    # Ensure indices are not negatives and in correct order
    if low < 0 or low >= high:
        return
    # Partition array and get the pivot index
    p = partition_lomuto(arr, low, high, use_last)
    # Sort the two partitions
    quick_sort_lomuto(arr, low, p - 1, use_last)  # Left side of pivot
    quick_sort_lomuto(arr, p + 1, high, use_last)  # Right side of pivot


# ======================================================================================================================
#                                          Hoare partition scheme
# ======================================================================================================================
def partition_hoare(arr: list, low: int, high: int) -> int:
    # Pivot value from the middle of the array
    pivot = arr[(high + low) // 2]
    # Initiate pivot indices
    i = low - 1  # Left index
    j = high + 1  # Right index

    while True:
        # Move the left index to the right
        i = i + 1
        while arr[i] < pivot:
            i = i + 1

        # Move the right index to the left at least
        j = j - 1
        while arr[j] > pivot:
            j = j - 1

        # If the indices crossed, return
        if i >= j:
            return j

        # Swap the elements at the left and right indices
        arr[i], arr[j] = arr[j], arr[i]


def quick_sort_hoare(arr: list, low: int, high: int) -> None:
    if len(arr) <= 1:
        return
    # Ensure indices are not negatives and in correct order
    if low < 0 or low >= high:
        return
    # Partition array and get the pivot index
    p = partition_hoare(arr, low, high)
    # Sort the two partitions
    quick_sort_hoare(arr, low, p)  # Left side of pivot
    quick_sort_hoare(arr, p + 1, high)  # Right side of pivot


# ======================================================================================================================
#                                                       Tests
# ======================================================================================================================
def is_sorted(arr: list) -> bool:
    for i in range(len(arr) - 1):
        if arr[i] > arr[i + 1]:
            return False
    return True


def quick_sort_test(func: callable, **kwargs) -> None:
    max_size = 500
    number_of_test = 100
    # We test for edge cases
    degenerated_arrays = [
        [],
        [1],
        [1, 1],
        [1, 2],
        [2, 1],
        list(range(max_size)),
        list(reversed(range(max_size))),
    ]
    for arr in degenerated_arrays:
        n = len(arr)
        func(arr, 0, n - 1, **kwargs)
        assert len(arr) == n
        assert is_sorted(arr)

    # We test random arrays
    for _ in range(number_of_test):
        arr = list(np.random.randint(max_size, size=max_size))
        n = len(arr)
        func(arr, 0, n - 1, **kwargs)
        assert len(arr) == n
        assert is_sorted(arr)

    print('TEST PASSED')


# ======================================================================================================================
#                                                      Time complexity
# ======================================================================================================================
def time_quick_sort(func: callable, arr: list, **kwargs) -> float:
    start = time.time()
    func(arr, 0, len(arr) - 1, **kwargs)
    end = time.time()
    return (end - start) * 1e3


def plot_time_complexity(func: callable, title: str, **kwargs):
    max_size = 5000
    step = int(max_size / 20)
    rng = range(0, max_size + step, step)

    # We need to increase the python recursion limit to max_size
    sys.setrecursionlimit(max_size * 2)

    times = {
        'unordered': [],
        'ordered': [],
        'reversed': [],
        'uniform': [],
    }

    # Random array
    for n in rng:
        arr = list(np.random.randint(n, size=n))
        t = time_quick_sort(func, arr,**kwargs)
        times['unordered'].append(t)

    max_yaxis = times['unordered'][-1] * 10

    # Degenerated arrays
    for n in rng:
        degenerated_arrays = {'ordered': list(range(n)),
                              'reversed': list(reversed(range(n))),
                              'uniform': [1] * n}
        for name, arr in degenerated_arrays.items():
            length = len(times[name])
            if length > max_size / step:
                continue
            if length == 0 or times[name][-1] != np.Inf:
                t = time_quick_sort(func, arr, **kwargs)
                # It will exit the graph
                if t > max_yaxis:
                    times[name].append(t)
                    t = np.Inf
            else:
                t = np.Inf
            times[name].append(t)

    df = pd.DataFrame(times, rng)
    fig = df.plot(title=title, labels=dict(index='array size', value='time', variable='array type'))
    fig.update_layout(yaxis=dict(range=[0, max_yaxis]))
    fig.update_yaxes(ticksuffix=' ms')
    fig.show()


if __name__ == '__main__':
    # Test
    quick_sort_test(quick_sort_lomuto, use_last=True)
    quick_sort_test(quick_sort_lomuto, use_last=False)
    quick_sort_test(quick_sort_hoare)

    # Plot
    plot_time_complexity(quick_sort_lomuto, title='Quick Sort - Lomuto', use_last=True)
    plot_time_complexity(quick_sort_lomuto, title='Quick Sort - Lomuto with median estimate', use_last=False)
    plot_time_complexity(quick_sort_hoare, title='Quick Sort - Hoare')

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
