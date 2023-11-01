def solution(my_string):
    str = []
    suffix = []
    suffix.append(my_string)
    for i in range(1, len(my_string)):
        str = my_string[len(my_string)-i:len(my_string):]
        suffix.append(str)
    return sorted(suffix)

print(solution("banana")) # ["a", "ana", "anana", "banana", "na", "nana"]