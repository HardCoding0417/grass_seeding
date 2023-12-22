def solution(n_str):
    str_list = list(n_str)
    i = 0
    while i < len(str_list) and str_list[i] == '0':
        str_list.pop(i)
    return ''.join(str_list)

print(solution('00020'))