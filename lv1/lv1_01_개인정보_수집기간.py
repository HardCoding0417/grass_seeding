from datetime import datetime, timedelta

today = ["2022.05.19"]
terms = ["A 6", "B 12", "C 3"]
privacies = ["2021.05.02 A", "2021.07.01 B", "2022.02.19 C", "2022.02.20 C"]

def solution(today, terms, privacies):
    answer = []
    dates = []
    letters = []
    term_dict = {}

    # 오늘 날짜를 int화
    today = int(today[0].replace('.', ''))

    # 계약일도 int화, 레터도 분리
    for privacie in privacies:
        date, letter = privacie.split(' ')
        date = date.replace('.', '')
        letters.append(letter)
        dates.append(int(date))

    # 약관을 딕셔너리로 만듬
    for term in terms:
        term_letter, term_num = term.split()
        term_dict[term_letter] = int(term_num)

    # 개인정보를 파기해야할지 말지 알려주는 함수
    # privacies + term을 했는데 today를 지났으면 파기하라는 메세지를 출력

    # privacies가 a인지 b인지 c인지 체크
    # a면 6개월을 추가. today를 넘었으면 파기를 추천


    for date, letter in zip(dates, letters):
        term = term_dict[letter] * 100  # 약관 기간 (월 단위를 일 단위로 환산)
        if date + term < today:
            answer.append(f"{date}: {letter} 정보를 파기해야 합니다.")
        else:
            answer.append(f"{date}: {letter} 정보를 파기할 필요가 없습니다.")

    return answer

print(solution(today, terms, privacies))


