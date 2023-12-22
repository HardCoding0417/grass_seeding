# 정수 배열 date1과 date2가 주어집니다.
# 두 배열은 각각 날짜를 나타내며 [year, month, day] 꼴로 주어집니다.
# 각 배열에서 year는 연도를, month는 월을, day는 날짜를 나타냅니다.
#
# 만약 date1이 date2보다 앞서는 날짜라면 1을, 아니면 0을 return 하는 solution 함수를 완성해 주세요.

def solution(date1, date2):
    for i in range(3):
        if date1[i] < date2[i]:
            return 1
        elif date1[i] > date2[i]:
            return 0
    return 0

date1 = [2021, 12, 1]
date2 = [2021, 11, 30]
print(solution(date1, date2))



# def solution(date1, date2):
#     return int(date1 < date2)
# 리스트끼리 부등호 비교도 되는구나...