import os

"""
PROBLEM:
Consider all the palindromes that can be constructed from some of the letters from s[l:r].
You can reorder the letters as you need. Some of these palindromes have the maximum length among all these palindromes.
How many maximum-length palindromes are there?


MODULAR FACTORIAL:
If p is prime, how to efficiently compute n! % p

A Naive Solution is to first compute n!, then compute n! % p. This solution works fine when the value of n! is small. 
The value of n! % p is generally needed for large values of n when n! cannot fit in a variable, and causes overflow. 

A Simple Solution is to one by one multiply result with i under modulo p. 
So the value of result doesn’t go beyond p before next iteration.
Time Complexity of this solution is O(n)

def modFact(n, p):
    if n >= p:
        return 0   
 
    result = 1
    for i in range(1, n + 1):
        result = (result * i) % p
 
    return result
    
Multiplicative Inverses
https://blogarithms.github.io/articles/2019-01/fermats-theorem


Binary Exponentiation
https://cp-algorithms.com/algebra/binary-exp.html
https://www.geeksforgeeks.org/modular-exponentiation-python/
"""


def initialize(string):
    n = len(string)
    z = ord('a')
    char_count = [[0] * LETTERS_NUMBER for _ in range(n + 1)]  # Do not copy array (cf. [[0] * LETTERS_NUMBER]*(n + 1))
    modular_fact = [1] * n
    modular_inv_fact = [1] * n

    # 2D array of sliding character count
    for i, char in enumerate(string, 1):
        for j in range(LETTERS_NUMBER):
            char_count[i][j] = char_count[i - 1][j] + (j == ord(char) - z)

    for i in range(1, n):
        modular_fact[i] = modular_fact[i - 1] * i % M  # modular factorial
        modular_inv_fact[i] = pow(modular_fact[i], M - 2, M)  # modular inverse of factorial using binary exponentiation

    return char_count, modular_fact, modular_inv_fact


def answerQuery(l, r):
    c, s, d = 0, 0, 1
    for j in [CHAR_COUNT[r][i] - CHAR_COUNT[l - 1][i] for i in range(LETTERS_NUMBER)]:
        c += j % 2  # count of center characters
        s += j // 2  # count of side characters
        d *= MODULAR_INV_FACT[j // 2]  # "denominators"
    return (c or 1) * MODULAR_FACT[s] * d % M


if __name__ == '__main__':
    string = 'madamimadam'
    l = 4
    r = 7
    LETTERS_NUMBER, M = 26, 1000000007
    CHAR_COUNT, MODULAR_FACT, MODULAR_INV_FACT = initialize(string)
    result = answerQuery(l, r)
    print(result)
