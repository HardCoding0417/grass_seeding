def solution(numbers, direction):
    if direction == 'right':
        last_num = numbers.pop(-1)
        numbers.insert(0, last_num)
        return numbers
    elif direction == 'left':
        first_num = numbers.pop(0)
        numbers.append(first_num)
        return numbers


print(solution([1, 2, 3], 'left'))

# 회전 라이브러리
# from collections import deque
#
# def solution(numbers, direction):
#     numbers = deque(numbers)
#     if direction == 'right':
#         numbers.rotate(1)
#     else:
#         numbers.rotate(-1)
#     return list(numbers)