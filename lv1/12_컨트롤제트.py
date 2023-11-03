def solution(s):
    numbers = s.split()  # 공백을 기준으로 숫자와 문자를 분리
    answer = 0
    prev_num = 0  # 'Z' 바로 이전의 숫자를 저장할 변수

    for num in numbers:
        if num != 'Z':
            answer += int(num)
            prev_num = int(num)  # 현재 숫자를 prev_num에 저장
        else:
            answer -= prev_num  # 'Z'를 만나면 이전 숫자를 뺌

    return answer

print(solution("1 2 Z 3"))