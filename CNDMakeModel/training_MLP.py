import os
import random

import cv2  # working with, mainly resizing, images
import matplotlib.pyplot as plt
import numpy as np  # dealing with arrays
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers.core import Dense, Activation
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.layers.core import Dropout

IMG_PATH = 'C:\\Users\\Greg\\Desktop\\Projets\\Dataset\\'
IMG_SIZE = 75
EPOCHS = 120
BS = 500
CLASSES = 2

train_data = []
train_labels = []
test_data = []

MODEL_NAME = 'C:\\Users\\lamyg\\Desktop\\Projets\\CatzNDogz\\CNDMakeModel\\CatzNDogz_mlp.model'


def get_data():
    train_dir = IMG_PATH + 'train'
    test_dir = IMG_PATH + 'test'

    train_dogs = [(train_dir + '/{}').format(i) for i in os.listdir(train_dir) if 'dog' in i]
    train_cats = [(train_dir + '/{}').format(i) for i in os.listdir(train_dir) if 'cat' in i]

    test_images = [(test_dir + '/{}').format(i) for i in os.listdir(test_dir)]

    train_images = train_dogs[:4000] + train_cats[:4000]
    random.shuffle(train_images)
    random.shuffle(test_images)

    create_data(train_images, test_images)


def create_data(train_images, test_images):

    for image in test_images:
        test_data.append(cv2.resize(cv2.imread(image, cv2.IMREAD_COLOR), (IMG_SIZE, IMG_SIZE)))

    for img in train_images:
        train_data.append(cv2.resize(cv2.imread(img, cv2.IMREAD_COLOR), (IMG_SIZE, IMG_SIZE)))

        if 'dog' in img:
            train_labels.append(1)
        elif 'cat' in img:
            train_labels.append(0)

    np.save('train_data_mlp.npy', np.array(train_data))
    np.save('train_labels_mlp.npy', np.array(train_labels))
    np.save('test_images_mlp.npy', np.array(test_data))


def make_model(dimData):
    model = Sequential()

    model.add(Dense(1024, input_shape=(dimData,)))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))

    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))

    model.add(Dense(CLASSES))
    model.add(Activation("softmax"))

    model.compile(loss="categorical_crossentropy", optimizer='adam',
                  metrics=["accuracy"])

    return model


def train_model():
    train_data = np.load('train_data.npy')
    train_labels = np.load('train_labels.npy')
    dimData = np.prod(train_data.shape[1:])

    k = train_data.shape[0]
    train_data = train_data.reshape(k, dimData)

    train_data = train_data.astype('float32')
    train_data /= 255

    (train_x, test_x, train_y, test_y) = train_test_split(train_data, train_labels, test_size=0.20, random_state=2)

    train_y = to_categorical(train_y)
    test_y = to_categorical(test_y)

    H = model = make_model(dimData)

    model.fit(train_x, train_y, batch_size=BS, validation_data=(test_x, test_y), epochs=EPOCHS, verbose=1)

    model.save(MODEL_NAME)

    plt.style.use("ggplot")
    plt.figure()
    N = EPOCHS
    plt.plot(np.arange(0, N), H.history.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on Cat/Dog")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig("plot_mlp")


def predict():
    x_test = np.load('test_images.npy')[0:10]
    x_test = x_test.astype('float32')
    x_test /= 255
    dimData = np.prod(x_test.shape[1:])
    k = x_test.shape[0]
    x_test = x_test.reshape(8000, dimData)


    model = load_model(MODEL_NAME)

    i=0
    text_labels = []

    plt.figure(figsize=(IMG_SIZE, IMG_SIZE))

    for batch in x_test:
        pred = model.predict(batch)

        if pred > 0.5:
            text_labels.append('dog')
        else:
            text_labels.append('cat')

        plt.subplot(4, 4, i +1)
        plt.title('This is a ' + text_labels[i])
        imgplot = plt.imshow(batch[0])
        i += 1

        if i % 10 == 0:
            break

    plt.show()


#get_data()



#train_model()

predict()
