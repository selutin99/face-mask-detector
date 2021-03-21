import os

from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename, redirect

from service import UPLOAD_FOLDER, allowed_file, test_single_image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('prediction', filename=filename))

    return redirect(url_for('index'))


@app.route('/prediction/<filename>')
def prediction(filename):
    full_name = UPLOAD_FOLDER + filename
    image_result = test_single_image(full_name)
    return render_template(
        'result.html',
        image='/' + full_name,
        main_prediction='/' + image_result
    )


if __name__ == '__main__':
    app.run()
