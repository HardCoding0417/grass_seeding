# 입력 받은 문자열에 a~Z에 속하는 문자가 몇 개인지 세어
# 52개의 요소를 가진 희소행렬을 출력
# 맵핑이 핵심

def solution(my_string):
    answer = [0 for i in range(52)]
    for string in my_string:
        if string.isupper():
            i = 65
        else:
            i = 97-26
        answer[ord(string) - i] += 1
    return answer

# def solution(my_string):
#     answer = [0 for i in range(52)]
#     for string in my_string:
#         if string.isupper():
#             k = 65
#         else:
#             k = 71
#         answer[ord(string) - k] += 1
#     return answer