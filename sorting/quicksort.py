import sys
import numpy as np
import time
import pandas as pd

pd.options.plotting.backend = "plotly"


# Runtime:
#           average: O(n log(n))
#           worst case: O(n^2) (by using lomuto algo if partitioned elements are far from the median)
#
# Memory: O(log(n) with Tail Call optimisation, O(n) otherwise)
#
# QuickSort is a Divide and Conquer algorithm. It's an inplace algorithm.
# It picks an element as pivot and partitions the given array around the picked pivot.

# ======================================================================================================================
#                                                   Trace Decorator
# ======================================================================================================================
class Trace:
    """
    Trace Decorator used to get the call stack depth
    """

    def __init__(self, func):
        self.func = func
        self.depth = 0

    def __call__(self, *args, **kwargs):
        self.func(*args, **kwargs)
        self.depth += 1

    def reset_trace(self):
        self.depth = 0


# ======================================================================================================================
#                                             Median of three
# ======================================================================================================================
def median_of_three(arr: list, low: int, high: int) -> None:
    """
    Fast estimate of the median
    """
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
    """
    The Lomuto partition scheme used to partition the array
    """
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


@Trace
def quicksort_lomuto(arr: list, low: int, high: int, use_last: bool) -> None:
    """
      The quicksort algo based on the Lomuto partition scheme
    """
    if len(arr) <= 1:
        return
    # Ensure indices are in correct order
    if low >= high:
        return
    # Partition array and get the pivot index
    p = partition_lomuto(arr, low, high, use_last)
    # Sort the two partitions
    quicksort_lomuto(arr, low, p - 1, use_last)  # Left side of pivot
    quicksort_lomuto(arr, p + 1, high, use_last)  # Right side of pivot


# ======================================================================================================================
#                                          Hoare partition scheme
# ======================================================================================================================
def partition_hoare(arr: list, low: int, high: int) -> int:
    """
    The Hoare partition scheme used to partition the array
    """
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


@Trace
def quicksort_hoare(arr: list, low: int, high: int) -> None:
    """
      The quicksort algo based on the Hoare partition scheme
    """
    if len(arr) <= 1:
        return
    # Ensure indices are in correct order
    if low >= high:
        return
    # Partition array and get the pivot index
    p = partition_hoare(arr, low, high)
    # Sort the two partitions
    quicksort_hoare(arr, low, p)  # Left side of pivot
    quicksort_hoare(arr, p + 1, high)  # Right side of pivot


# ======================================================================================================================
#                                          Tail call optimisation (TCO)
# ======================================================================================================================
@Trace
def quicksort_tco(arr: list, low: int, high: int) -> None:
    """
      The quicksort algo based on the Hoare partition scheme with Tail call optimisation (TCO)
    """
    if len(arr) <= 1:
        return

    while low < high:
        p = partition_hoare(arr, low, high)

        # If left is smaller we recur on left and iterate on right
        if p - low < high - p:
            if low < p:  # Avoid the last call to reduce the call stack
                quicksort_tco(arr, low, p)
            low = p + 1
        else:
            if p + 1 < high:  # Avoid the last call to reduce the call stack
                quicksort_tco(arr, p + 1, high)
            high = p


# ======================================================================================================================
#                                                   Tests
# ======================================================================================================================
def is_sorted(arr: list) -> bool:
    """
    Test if the array is sorted
    """
    for i in range(len(arr) - 1):
        if arr[i] > arr[i + 1]:
            return False
    return True


def quicksort_test(func: callable, **kwargs) -> None:
    """
    Test a sorting function against multiple arrays
    """
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
#                                             Time and space complexity
# ======================================================================================================================
def time_and_depth(func: callable, arr: list, **kwargs) -> tuple[float, float]:
    """
    Compute the time and recursion depth of a Quicksort algo
    """
    func.reset_trace()
    start = time.time()
    func(arr, 0, len(arr) - 1, **kwargs)
    end = time.time()
    return (end - start) * 1e3, func.depth


def plot_time_complexity(func: callable, title: str, **kwargs):
    """
      Plot the time needed to sort arrays of different size and type
    """
    max_size = 2000
    step = int(max_size / 20)
    rng = range(0, max_size + step, step)

    # We need to increase the python recursion limit to max_size
    sys.setrecursionlimit(max_size * 3)

    array_types = ['unsorted',
                   'sorted',
                   'reverse-sorted',
                   'equal-elements']

    times = {arr_type: [] for arr_type in array_types}

    # Random array
    for n in rng:
        arr = list(np.random.randint(n, size=n))
        t, _ = time_and_depth(func, arr, **kwargs)
        times['unsorted'].append(t)

    max_time_yaxis = times['unsorted'][-1] * 10

    # Degenerated arrays
    for n in rng:
        degenerated_arrays = {'sorted': list(range(n)),
                              'reverse-sorted': list(reversed(range(n))),
                              'equal-elements': [1] * n}
        for name, arr in degenerated_arrays.items():
            length = len(times[name])
            if length > max_size / step:
                continue
            if length == 0 or times[name][-1] != np.Inf:
                t, _ = time_and_depth(func, arr, **kwargs)
                # It will exit the graph
                if t > max_time_yaxis:
                    times[name].append(t)
                    t = np.Inf
            else:
                t = np.Inf
            times[name].append(t)

    # Plot Time
    df = pd.DataFrame(times, rng)
    fig = df.plot(title=title, labels=dict(index='array size', value='time', variable='array type'))
    fig.update_layout(yaxis=dict(range=[0, max_time_yaxis]))
    fig.update_yaxes(ticksuffix=' ms')
    fig.show()


def plot_recursion_depth():
    """
      Plot the recursion call depth of Quicksort with the Hoarse Partition scheme
      with and without call tail optimisation
    """
    max_size = 5000
    step = int(max_size / 20)
    rng = range(0, max_size + step, step)

    # We need to increase the python recursion limit to max_size
    sys.setrecursionlimit(max_size * 3)

    depths = {'without_tco': [],
              'with_tco': []}

    # Random array
    for n in rng:
        arr = list(np.random.randint(n, size=n))
        arr_copy = arr.copy()
        _, depth_without_tco = time_and_depth(quicksort_hoare, arr)
        _, depth_with_tco = time_and_depth(quicksort_tco, arr_copy)
        depths['without_tco'].append(depth_without_tco)
        depths['with_tco'].append(depth_with_tco)

    # Plot depths
    df = pd.DataFrame(depths, rng)
    fig = df.plot(title='Recursion Depth', labels=dict(index='array size',
                                                       value='Recursion depth',
                                                       variable='array type'))
    fig.show()


# ======================================================================================================================
#                                                     Run
# ======================================================================================================================
if __name__ == '__main__':
    # Test
    quicksort_test(quicksort_lomuto, use_last=True)
    quicksort_test(quicksort_lomuto, use_last=False)
    quicksort_test(quicksort_hoare)
    quicksort_test(quicksort_tco)

    # Plot Time
    plot_time_complexity(quicksort_lomuto, title='Quick Sort - Lomuto', use_last=True)
    plot_time_complexity(quicksort_lomuto, title='Quick Sort - Lomuto with median estimate', use_last=False)
    plot_time_complexity(quicksort_hoare, title='Quick Sort - Hoare')
    plot_time_complexity(quicksort_tco, title='Quick Sort - Hoare with TCO')

    # Plot Recursion Depth
    plot_recursion_depth()

# ======================================================================================================================
#                                                  Example
# ======================================================================================================================

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
