# def solution(num_list, n):
#     answer = []
#     for i in range(1, len(num_list)//n+2):
#         segment = (num_list[i*n-n:i*n])
#         if segment:
#             answer.append(segment)
#     return answer

import numpy as np
def solution(num_list, n):
    li = np.array(num_list).reshape(-1,n)
    return li.tolist()

print(solution([1, 2, 3, 4, 5, 6, 7, 8], 3))
