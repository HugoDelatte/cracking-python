# Check if a string is a subsequence of another string ( using Stacks )

def check_if_subsequence(string: str, target: str):
    stack = list(target)
    for i in reversed(range(len(string))):
        if len(stack) == 0:
            return 'YES'
        if string[i] == stack[-1]:
            del stack[-1]
    if len(stack) == 0:
        return 'YES'
    else:
        return 'NO'


if __name__ == '__main__':
    check_if_subsequence('ABCDEFG', 'CEG')
    check_if_subsequence('ABCDEFG', 'CEA')
