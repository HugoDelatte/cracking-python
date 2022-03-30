# Permutations --> order matter
# Combinations --> order doesn't matter

# nPk =  n! / (n-k)!
# Ex: permutations of 3 elements in {1, 2, 3}: (1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1)


import itertools


def permutations(arr: iter):
    if len(arr) <= 1:
        yield tuple(arr)
    else:
        for p in permutations(arr[1:]):
            for i in range(len(arr)):
                yield *p[:i], arr[0], *p[i:]


def k_permutations(arr: iter, k: int):
    n = len(arr)
    for indices in itertools.product(range(n), repeat=k):
        if len(set(indices)) == k:
            yield tuple(arr[i] for i in indices)


if __name__ == '__main__':
    a = range(5)
    print(list(permutations(a)))
    print(list(itertools.permutations(a)))
    print(list(k_permutations(a, 2)))
