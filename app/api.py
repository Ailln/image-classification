from flask import Flask
from flask import request

from app import test

app = Flask(__name__)


@app.route("/api")
def index():
    return "Please use /api/image_classification."


@app.route('/api/image_classification', methods=['GET', 'POST'])
def text_classification():
    if request.method == 'POST':
        img = request.files.get('upload')
        save_path = f"./save/{img.filename}"
        img.save(save_path)
        img_class = test.test(save_path)
        print(f">> result: {img_class}")
        return img_class
    else:
        return "Please use POST method."


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8000)
