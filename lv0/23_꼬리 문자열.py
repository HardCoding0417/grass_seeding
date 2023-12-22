# 문자열 리스트 str_list와 제외하려는 문자열 ex가 주어질 때,
# str_list에서 ex를 포함한 문자열을 제외하고 만든 꼬리 문자열을 return하도록 solution 함수를 완성해주세요.

# def solution(str_list, ex):
#     answer = [i.replace(ex, '') if ex in i else i for i in str_list]
#     return ''.join(answer)
#
# str_list = ["abc", "def", "ghi"]
# ex = "ef"
#
# print(solution(str_list, ex))
# 문제를 잘못 이해했다. ex만 제외하고 합치는 줄 알았는데 그게 아니었다.


def solution(str_list, ex):
    i = len(str_list) - 1
    while i >= 0:
        if ex in str_list[i]:
            str_list.pop(i)
        i -= 1
    return ''.join(str_list)

str_list = ["abc", "def", "ghi"]
ex = "ef"

print(solution(str_list, ex))

# 인덱스 변경 없이 요소 제거하는 법을 연습하기 위해
# 리스트 컴프리헨션이 아닌 방법으로 해보았다.
