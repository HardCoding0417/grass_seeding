def solution(my_string, is_suffix):
    str = []
    suffix = []
    suffix.append(my_string)
    for i in range(1, len(my_string)):
        str = my_string[len(my_string)-i:len(my_string):]
        suffix.append(str)
    suffix = sorted(suffix)

    for i in suffix:
        if is_suffix == i:
            return 1
    return 0

print(solution("banana", 'ana')) # ["a", "ana", "anana", "banana", "na", "nana"]