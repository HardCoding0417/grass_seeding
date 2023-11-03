# def solution(my_string):
#     my_string = [i for i in my_string if not(i in 'aeiou')]
#
#     return ''.join(my_string)
# print(solution('nice to meet you'))

# def solution(my_string):
#     return sorted([int(i) for i in my_string if (i in '123456789')])
#
# print(solution('r1qwe124124'))


def solution(my_string):
    num_list = [int(i) for i in my_string if i in '0123456789']
    return sum(num_list)

print(solution('r1qwe124124'))
