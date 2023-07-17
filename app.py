from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def logo():
    return render_template('logo.html')

@app.route('/ocr')
def ocr():
    return render_template('ocr.html')

@app.route('/index')
def index():
    return render_template('index.html')

posts = [
    {"title": "글 제목 1", "content": "글 내용 1", "image": "/static/images/image1.jpg"},
    {"title": "글 제목 2", "content": "글 내용 2", "image": "/static/images/image2.jpg"},
    {"title": "글 제목 3", "content": "글 내용 3", "image": "/static/images/image3.jpg"},
    {"title": "글 제목 4", "content": "글 내용 4", "image": "/static/images/image4.jpg"},
    {"title": "글 제목 5", "content": "글 내용 5", "image": "/static/images/image5.jpg"}
]

@app.route('/main')
def main():
    return render_template('main.html',posts=posts)

if __name__ == '__main__':
    app.run()
