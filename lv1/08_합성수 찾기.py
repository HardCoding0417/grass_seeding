# n 이하의 숫자 중에 합성수가 몇 개인지 찾아내기

# from sympy import isprime, primerange, nextprime
#
# def solution(n):
#     answer = []
#     for i in range(2, n+1):
#         if not isprime(i):
#             answer.append(i)
#     return answer
#

# 에라토스테네스의 체
# 2로 나눠지는 수들을 없애고
# 3으로 나눠지는 수들을 없애고... 를 n까지 반복

# def solution(n):
#
#     answer = [False,False] + [True]*(n-1)
#     primes=[]
#
#     for i in range(2,n+1):
#       if answer[i]:
#         primes.append(i)
#         for j in range(2*i, n+1, i):
#             answer[j] = False
#     print(primes)
#
# solution(1000)


# 합성수 찾기.
def solution(n):
    sieve = [True] * (n+1)  # 모든 숫자를 소수로 가정합니다.
    m = int(n ** 0.5)  # n의 제곱근까지만 확인합니다.
    for i in range(2, m+1):
        if sieve[i] is True:  # i가 소수인 경우
            for j in range(i*i, n+1, i):  # i의 배수들을 False로 설정합니다.
                sieve[j] = False

    # sieve 리스트에서 True인 인덱스만 추출합니다. 이 인덱스들이 소수입니다.
    primes = [i for i in range(2, n+1) if not sieve[i]]
    print(primes)
solution(1000)


def solution(n):
    sieve = [True] * (n+1)
    m = int(n ** 0.5)
    for i in range(2, m+1):
        if sieve[i] is True:
            for j in range(i*i, n+1, i):
                sieve[j] = False

    primes = [i for i in range(2, n+1) if not sieve[i]]
    print(primes)
solution(1000)