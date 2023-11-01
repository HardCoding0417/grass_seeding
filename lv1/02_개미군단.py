def solution(hp):
    ant = {'general' : 5,
           'soldier' : 3,
           'worker' : 1}
    katydid = { 'hp' : 1 }
    katydid['hp'] = hp
    first_value, first_remainder = divmod(katydid['hp'], ant['general'])
    second_value, second_remainder = divmod(first_remainder, ant['soldier'])
    return first_value + second_value + second_remainder