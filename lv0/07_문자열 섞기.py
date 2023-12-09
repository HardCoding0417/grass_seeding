

str1 = "ㅎㅇ"
str2 = '하이하이'

def solution(str1, str2):
    answer = []
    combined_str = []
    for i in zip(str1, str2):
        combined_str.extend(i)
        answer = ''.join(combined_str)
    return answer

print(solution(str1, str2))
