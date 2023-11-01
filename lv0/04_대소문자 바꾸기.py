str = 'aBcDeFg'
answer = []
for i in str:
    if i.islower() is True:
        answer.append(i.upper())
    if i.isupper() is True:
        answer.append(i.lower())
result = ''.join(answer)
print(result)