def solution(array):
    n = len(array)
    if n % 2 == 1:
        return array[n // 2]
    else:
        return array[(n - 1) // 2]

print(solution([9, -1, 0]))
