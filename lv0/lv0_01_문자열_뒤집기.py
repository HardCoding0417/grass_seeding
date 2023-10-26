def solution(my_string, s, e):
    index_string = my_string[s:e+1]
    answer = my_string[:s] + index_string[::-1] + my_string[e+1:]
    return answer

a = solution('Progra21Sremm3.',6,12)
print(a)