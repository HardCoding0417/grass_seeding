def solution(intStrs, k, s, l):
    answers = []
    for i in range(len(intStrs)):
        answer = intStrs[i]
        answer = answer[s:s+l]
        answer = int(answer)
        if answer > k:
            answers.append(answer)
    return answers
print(solution(["0123456789","9876543210","9999999999999"],50000,5,5))
