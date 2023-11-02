#
# def solution(my_string, indices):
#     answer = []
#     my_string = list(my_string)
#     for i in indices:
#         a = my_string.pop(i)
#         answer.append(a)
#     return ''.join(answer)
#



def solution(my_string, indices):
    return ''.join([char for i, char in enumerate(my_string) if i not in indices])

print(solution("apporoograpemmemprs", [1, 16, 6, 15, 0, 10, 11, 3]))
