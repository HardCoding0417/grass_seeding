def solution(my_string):
    my_string = my_string.replace(" ", "")
    num = ''
    numbers = []
    operators = []

    # 숫자와 연산자 분리
    for char in my_string:
        if char in '+-':
            numbers.append(int(num))
            operators.append(char)
            num = ''
        else:
            num += char
    numbers.append(int(num))

    # 계산
    result = numbers[0]
    for i in range(len(operators)):
        if operators[i] == '+':
            result += numbers[i + 1]
        elif operators[i] == '-':
            result -= numbers[i + 1]

    return result


# 연산자와 숫자는 반드시 짝지어져 있다는 것을 의식한 함수


# def solution(my_string):
#     return sum(int(i) for i in my_string.replace(' - ', ' + -').split(' + '))
# -를 +-로 만들어서 스플릿한 뒤 sum으로 더한 함수.
# 기발하다



my_string = "351 + 235 - 32 + 14"
print(solution(my_string))


