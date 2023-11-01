def solution(my_strings, parts):
    answer = []
    for i in range(len(parts)):
        my_string = my_strings[i]
        s, e = parts[i]
        answer.append(my_string[s:e+1])
    return ''.join(answer)

print(solution(["progressive", "hamburger", "hammer", "ahocorasick"], [[0, 4], [1, 2], [3, 5], [7, 7]]))


