def solution(number, n, m):
    if ((number % n) == 0) and ((number % m) == 0):
        return 1
    else:
        return 0

print(solution(60, 2, 3))

# 이 문제를 포함해서, 프로그래머스에서 추천한 문제 10개를 풀었음.