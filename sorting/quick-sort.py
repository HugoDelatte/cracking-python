import typing

# QuickSort is a Divide and Conquer algorithm.
# It picks an element as pivot and partitions the given array around the picked pivot.
#
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
