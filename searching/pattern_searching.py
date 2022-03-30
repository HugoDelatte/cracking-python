# Algorithm for Pattern Searching

# Given a text txt[0..n-1] and a pattern pat[0..m-1], write a function that prints all occurrences of pat[] in txt[].

# Naive Pattern Searching
# Slide the pattern over text one by one and check for a match.
# If a match is found, then slides by 1 again to check for subsequent matches.
# Best case = O(n)
# Worst case = O(m*n)  (occurs when only the last character is different)
def naive_search(pattern: str, text: str):
    n = len(text)
    m = len(pattern)
    for i in range(n - m + 1):
        j = 0
        while j < m:
            if text[i + j] != pattern[j]:
                break
            j += 1
        if j == m:
            print(f'Pattern found at index : {i}')


# KMP Algorithm for Pattern Searching
# Worst case complexity to O(n).
# When we detect a mismatch, we already know some of the characters in the text of the next window.
# We take advantage of this information to avoid matching the characters that we know will anyway match
# We construct an auxiliary lps[] of size m (same as size of pattern) which is used to skip characters while matching.
# LPS = Longest Prefix Suffix
# lps[i] = the longest proper prefix of pat[0..i] which is also a suffix of pat[0..i]
# For the pattern “AABAACAABAA”, lps = [0, 1, 0, 1, 2, 0, 1, 2, 3, 4, 5]

def init_lps(pattern: str) -> list:
    lps = [0] * len(pattern)
    i = 1
    length = 0
    while i < len(pattern):
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length > 0:
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1
    return lps


def kmp_search(patter: list, text: str):
    lps = init_lps(patter)
    i = 0
    j = 0
    while i < len(text):
        if patter[j] == text[i]:
            i += 1
            j += 1
        if j == len(patter):
            print(f'Pattern found at index : {i - j}')
            j = lps[j - 1]
        elif i < len(text) and patter[j] != text[i]:
            if j > 0:
                j = lps[j - 1]
            else:
                i += 1


if __name__ == '__main__':
    txt = "AABAACAADAABAAABAA"
    pat = "AABA"
    naive_search(pat, txt)
    kmp_search(pat, txt)
