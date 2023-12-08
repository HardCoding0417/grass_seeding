from flask import Flask

app = Flask(__name__)

# 홈 화면 만듬
@app.route('/')
@app.route('/home')
def home():
    return 'Hello, Flask!'

# 유저 화면 만듬
@app.route('/user')
def user():
    return 'Hello User!'
# http://127.0.0.1:5000/user로 접근 가능

if __name__ == '__main__':
    app.run(debug=True)

