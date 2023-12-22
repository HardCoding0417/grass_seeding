# def solution(arr, delete_list):
#     for i in arr:
#         if i in delete_list:
#             arr.remove(i)
#     return arr

def solution(arr, delete_list):
    answer = []
    for i in arr:
        if i not in delete_list:
            answer.append(i)
    return answer

delete_list = [94, 777, 104, 1000, 1, 12]
arr = [293, 1000, 395, 678, 94]

delete_list2 = [377, 823, 119, 43]
arr2 = [110, 66, 439, 785, 1]

print(solution(arr2, delete_list2))