import cv2
import numpy as np  # dealing with arrays
import matplotlib.pyplot as plt
import os, time
from PIL import Image
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from flask import Flask, request, render_template, send_from_directory, flash, send_from_directory  # working with, mainly resizing, images
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing.image import img_to_array

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'jfif'])
IMG_SIZE = 75


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
        target = os.path.join(APP_ROOT, 'static')

        if not os.path.isdir(target):
            os.mkdir(target)

        model = request.files["model"]
        files = request.files.getlist("files")

        if files and model:
            modelname = model.filename
            filename = predict(files, modelname)

            return render_template('result.html', image_name=filename)

        else:
            flash('No file part')
            return render_template('index.html')


@app.route('/send_image/<name>')
def send_image(name):
    image = send_from_directory("static", name)
    return image


def predict(files, model):
    model = load_model(model)
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    i = 0
    labels = []
    plt.figure(figsize=(10, 10), dpi=80)
    plt.xlabel('xlabel', fontsize=18)
    plt.ylabel('ylabel', fontsize=16)
    plt.axis("off")
    for file in files:
        image = dataFromImage(file)
        img = img_to_array(image)
        img = np.expand_dims(image, axis=0)
        image = Image.fromarray(image, 'RGB')
        label = ''

        result = model.predict(test_datagen.flow(img, batch_size=1))

        if result > 0.5:
            labels.append('dog')
        else:
            labels.append('cat')

        plt.subplot(2, 2, i+1)
        plt.title('This is a ' + labels[i])
        imgplot = plt.imshow(image)
        i += 1

        if i % 10 == 0:
            break

    target = os.path.join(APP_ROOT, 'static')
    image_name = "plot" + str(time.time()) +".jpeg"
    finalUrl = "/".join([target, image_name])
    plt.savefig(finalUrl, bbox_inches="tight")

    return image_name


def dataFromImage(file):
    data = file.read()
    image = np.asarray(bytearray(data))
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    return image


@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-\
    revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response


if __name__ == "__main__":
    app.run()
