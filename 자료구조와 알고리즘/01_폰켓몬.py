# N마리 중에 N/2마리를 가져가도 됨
# [3, 1, 2, 3]이면 3번 폰켓몬이 2마리 있는 것
# 4마리 중 2마리를 고르는 경우의 수는 6개
# 최대한 다양한 폰켓몬을 많이 갖고 싶음

# N마리 폰켓몬의 종류 번호가 담긴 배열 nums가 매개변수로 주어질 때,
# N/2마리의 폰켓몬을 선택하는 방법 중,
# 가장 많은 종류의 폰켓몬을 선택하는 방법을 찾아,
# 그때의 '폰켓몬 종류 번호'의 개수를 return 하도록 solution 함수를 완성해주세요.


def solution(nums):
    num_len = len(set(nums))
    result = len(nums) // 2
    answer = min(result, num_len)
    return answer


print(solution([3,3,3,2,2,1,3,6,8,4,7,6,5,1,1,1,1,11,1,1,1,1,]))

