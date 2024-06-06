from flask import Flask, render_template, request, redirect
import os
from predict_from_microphone import predict_emotion_from_audio_file, predict_emotion_from_microphone, emotion_labels
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return redirect(request.url)
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                emotion_index, confidence = predict_emotion_from_audio_file(filepath)
                emotion = emotion_labels[emotion_index]
                return render_template('result.html', emotion=emotion, confidence=confidence)
        elif 'record' in request.form:
            emotion_index, confidence = predict_emotion_from_microphone()
            emotion = emotion_labels[emotion_index]
            return render_template('result.html', emotion=emotion, confidence=confidence)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
