import cv2
import numpy as np  # dealing with arrays
import os
from PIL import Image
from flask import Flask, request, render_template, send_from_directory, flash  # working with, mainly resizing, images
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing.image import img_to_array

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'jfif'])
IMG_SIZE = 75
MODEL_NAME = 'CatzNDogz.model'

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
# Config options - Make sure you created a 'config.py' file.
app.config.from_object('config')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/result', methods=['POST', 'GET'])
def result():
    if request.method == 'POST':
        target = os.path.join(APP_ROOT, 'static/')

        if not os.path.isdir(target):
            os.mkdir(target)

        if not request.files['file']:
            flash('No file part')
            return render_template('index.html')

        file = request.files['file']
        if file and allowed_file(file.filename):
            pet_name = predict(file)
            file.stream.seek(0)

            img = Image.open(file)
            wpercent = (150 / float(img.size[0]))
            hsize = int((float(img.size[0] * float(wpercent))))
            img = img.resize((150, hsize), Image.ANTIALIAS)
            destination = "/".join([target, file.filename])
            img.save(destination)

            return render_template('result.html', pet=pet_name, image_name=file.filename)

@app.route('/send_image/<filename>')
def send_image(filename):
    return send_from_directory("static", filename)


def predict(file):
    image = dataFromImage(file)
    model = load_model(MODEL_NAME)
    label = ''
    
    result = model.predict(image)
    
    if result > 0.5:
        label = 'dog'
    else:
        label = 'cat'

    return label


def dataFromImage(file):
    data = file.read()
    image = np.asarray(bytearray(data))

    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    return image


if __name__ == "__main__":
    app.run()
