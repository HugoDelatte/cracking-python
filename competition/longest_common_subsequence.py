# Longest Common Subsequenc (LCS)

# Given two sequences, find the length of longest subsequence present in both of them.
# A subsequence is a sequence that appears in the same relative order, but not necessarily contiguous.
# For example, “abc”, “abg”, “bdf”, “aeg”, ‘”acefg”, .. etc are subsequences of “abcdefg”

# Brute Force
#  The number of possible different subsequences of a string with length n is Cn0 + Cn1 + Cn2 + … Cnn = 2n-1.
# And it takes O(n) time to check if a subsequence is common to both the strings.
# So the complexity is O(n * 2^n)


# Dynamic programming
# Time Complexity = O(m*n)
# The idea is to find the length of the longest common suffix for all substrings of both strings and store these
# lengths in a table.

# if X[m-1] = Y[n-1]:
#       LCSuff(X, Y, m, n) = LCSuff(X, Y, m-1, n-1) + 1
# else:
#       LCSuff(X, Y, m, n) = 0

# LCSubStr(X, Y, m, n) = Max(LCSuff(X, Y, i, j)) where 1 <= i <= m and 1 <= j <= n


import math
import itertools


def lcs_v1(s1: str, s2: str):
    n = len(s1)
    m = len(s2)

    mat = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if s1[i - 1] == s2[j - 1]:
                mat[i][j] = mat[i - 1][j - 1] + 1
            else:
                mat[i][j] = max(mat[i - 1][j], mat[i][j - 1])
    return mat[n][m]


def lcs_v2(s1, s2, m, n):
    if m == 0 or n == 0:
        return 0
    elif s1[m - 1] == s2[n - 1]:
        return 1 + lcs_v2(s1, s2, m - 1, n - 1)
    else:
        return max(lcs_v2(s1, s2, m, n - 1), lcs_v2(s1, s2, m - 1, n))


if __name__ == '__main__':
    s1 = 'AGGTAB'
    s2 = 'GXTXAYB'
    lcs_v1(s1, s2)
    lcs_v2(s1, s2, len(s1), len(s2))
