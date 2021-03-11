#Usage: python app.py
import os
import flask
from flask import Flask, render_template, request, redirect, url_for
from flask import send_from_directory
from tensorflow.keras.models import load_model
# from keras.models import load_model
import numpy as np
import cv2
import time

MODEL_PATH = os.path.sep.join(["output", "flower.model"])
print("[INFO] loading model...")
model = load_model(MODEL_PATH)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])


def predict(file):
  image = cv2.imread(file)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  image = cv2.resize(image, (224, 224))

  image = image.astype("float32")
  mean = np.array([123.68, 116.779, 103.939][::-1], dtype="float32")
  image -= mean

  CLASSES = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]

  preds = model.predict(np.expand_dims(image, axis=0))[0]
  i = np.argmax(preds)
  label = CLASSES[i]

  return label

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def template_test():
    return render_template('template.html', label='Recomended CLASSES: daisy, dandelion, roses, sunflowers, tulips', imagesource='../uploads/template.png')

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        start_time = time.time()
        file = request.files['file']

        if file and allowed_file(file.filename):
            filename = file.filename
            print(filename)

            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            print(file_path)
            result = predict(file_path)

            os.rename(file_path, os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print("--- %s seconds ---" % str (time.time() - start_time))
            return render_template('template.html', label=result, imagesource='../uploads/' + filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

if __name__ == "__main__":
    # app.run()
    app.run(debug=True, host='0.0.0.0', port=5000)
