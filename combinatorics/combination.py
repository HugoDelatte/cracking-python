# Combinations --> order doesn't matter
# Permutations --> order matter

# nCk =  n! / (k! (n-k)!)
# nCk = nPk / k!
# sum_k(nCk) = 2^n

# Ex: combination of 2 elements (2-combination) in {1, 2, 3}: (1, 2), (1, 3), (2, 3)
# k-combination with repetitions is a sample of k elements from a set of n elements allowing for duplicates
# (i.e. with replacement) but disregarding different orderings
# nCk_with_repetitions = (n+k-1)Pk = (n+k-1)! / k!
# combinations('ABCD', 2) --> AB AC AD BC BD CD

import itertools


def combinations_v1(arr: iter, k: int):
    """
    Using recursion with list
    """
    if k > len(arr):
        return

    if k == 0:
        return [[]]

    res = []
    for i in range(len(arr)):
        m = arr[i]
        rem_arr = arr[i + 1:]
        for p in combinations_v1(rem_arr, k - 1):
            res.append([m] + p)
    return res


def combinations_v2(arr: iter, k: int):
    """
    Using recursion with generator
    """
    if k > len(arr):
        return

    if k == 0:
        yield ()

    for i in range(len(arr)):
        m = arr[i]
        rem_arr = arr[i + 1:]
        for p in combinations_v2(rem_arr, k - 1):
            yield m, *p


def combinations_v3(arr: iter, k: int):
    """
    Using iteration
    """
    pool = tuple(arr)
    n = len(pool)
    if k > n:
        return
    indices = list(range(k))
    yield tuple(pool[i] for i in indices)
    while True:
        for i in reversed(range(k)):
            if indices[i] != i + n - k:
                break
        else:
            return
        indices[i] += 1
        for j in range(i + 1, k):
            indices[j] = indices[j - 1] + 1
        yield tuple(pool[i] for i in indices)


def combinations_with_replacement(iterable, r):
    # combinations_with_replacement('ABC', 2) --> AA AB AC BB BC CC
    pool = tuple(iterable)
    n = len(pool)
    if not n and r:
        return
    indices = [0] * r
    yield tuple(pool[i] for i in indices)
    while True:
        for i in reversed(range(r)):
            if indices[i] != n - 1:
                break
        else:
            return
        indices[i:] = [indices[i] + 1] * (r - i)
        yield tuple(pool[i] for i in indices)


if __name__ == '__main__':
    a = range(5)
    print(combinations_v1(a, 3))
    print(list(combinations_v2(a, 3)))
    print(list(combinations_v3(a, 3)))
    print(list(itertools.combinations(a, 3)))
    print(list(combinations_with_replacement(a, 3)))
