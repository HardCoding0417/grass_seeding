from flask import Flask

app = Flask(__name__)

@app.route('/')
@app.route('/home')
def home():
    return '''
    <h1> 플라스크 홈페이지 URL</h1>
    <p> 이 페이지에는 플라스크 홈페이지로 가는 URL이 있습니다. </p>
    <a href="https://flask.palletsprojects.com">Flask 홈페이지 바로가기</a>
    '''


# 동적 웹페이지 만들기
@app.route('/user/<user_name>/<int:user_id>')
def user(user_name, user_id):
    return f'Hello, {user_name}({user_id})!'
# http://127.0.0.1:5000/user/me/1234 이렇게 접근할 시 Hello, me(1234)! 이런 메세지를 뱉는다.

if __name__ == '__main__':
    app.run(debug=True)