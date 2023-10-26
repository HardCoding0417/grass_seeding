def solution(my_string, overwrite_string, s):
    prefix = my_string[:s]
    fix_this = my_string[s + len(overwrite_string):]
    result = prefix + overwrite_string + fix_this
    return result

print(solution('안녕하세요 제 이름은 문정환입니다.', '3부터 덮어쓰겠습니다.', 3))

my_s = '안녕하세요 제 이름은 문정환입니다.'

print(my_s[3:])