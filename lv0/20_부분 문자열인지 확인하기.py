def solution(my_string, target):
    substrings = []
    for start in range(len(my_string)):
        for end in range(start + 1, len(my_string) + 1):
            substrings.append(my_string[start:end])
    if target in substrings:
        return 1
    else:
        return 0

# 1위의 해답
# def solution(my_string, target):
#     return int(target in my_string)
