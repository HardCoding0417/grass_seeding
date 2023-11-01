# letter가 매개변수로 주어질 때, letter를 영어 소문자로 바꾼 문자열을 return 하도록 solution 함수를 완성해보세요.
# "hello"

import re

def solution(letter):
    answer = []
    morse_dict = {
        '.-':'a','-...':'b','-.-.':'c','-..':'d','.':'e','..-.':'f',
        '--.':'g','....':'h','..':'i','.---':'j','-.-':'k','.-..':'l',
        '--':'m','-.':'n','---':'o','.--.':'p','--.-':'q','.-.':'r',
        '...':'s','-':'t','..-':'u','...-':'v','.--':'w','-..-':'x',
        '-.--':'y','--..':'z'}
    letter = letter.split(' ')
    for let in letter:
        word = morse_dict[let]
        answer.append(word)
    return ''.join(answer)
hello = ".... . .-.. .-.. ---"
print(solution(hello))