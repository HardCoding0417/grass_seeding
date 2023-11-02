# 문자열 my_string과 두 정수 m, c가 주어집니다.
# my_string을 한 줄에 m 글자씩 가로로 적었을 때
# 왼쪽부터 세로로 c번째 열에 적힌 글자들을
# 문자열로 return 하는 solution 함수를 작성해 주세요.

def solution(my_string, m, c):
    arr = []
    answer = []
    for i in range(1, len(my_string)//m+1):
        arr.append(my_string[(i-1)*m:m*i])
    print(arr)
    for i in arr:
        answer.append(i[c-1])
    return ''.join(answer)

print(solution("ihrhbakrfpndopljhygc",4,2))

