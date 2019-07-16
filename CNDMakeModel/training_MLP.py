import cv2  # working with, mainly resizing, images
import matplotlib.pyplot as plt
from tensorflow.python.keras.optimizers import Adam
import numpy as np  # dealing with arrays
from tensorflow.python.keras.layers.convolutional import Conv2D, MaxPooling2D
from tensorflow.python.keras.layers.core import Dense, Dropout, Flatten, Activation
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import os, random

IMG_PATH = 'C:\\Users\\xlmgr\\Desktop\\Docs\\Cours\\Dataset\\'
IMG_SIZE = 75
EPOCHS = 80
BS = 32
CLASSES = 1
INIT_LR = 1e-3

train_data = []
train_labels = []
test_data = []

MODEL_NAME = 'C:\\Users\\xlmgr\\PycharmProjects\\CNDMakeModel\\CatzNDogz.model'

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

    np.save('train_data.npy', np.array(train_data))
    np.save('train_labels.npy', np.array(train_labels))
    np.save('test_images.npy', np.array(test_data))


def make_model():
    model = Sequential()

    model.add(Dense(512))
    model.add(Activation("relu"))

    model.add(Dense(CLASSES))
    model.add(Activation("sigmoid"))

    opt = Adam(lr=INIT_LR)
    model.compile(loss="binary_crossentropy", optimizer=opt,
                  metrics=["accuracy"])

    return model


def train_model():
    (train_x, test_x, train_y, test_y) = train_test_split(train_data, train_labels, test_size=0.20, random_state=2)

    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2,
                             zoom_range=0.2, horizontal_flip=True)

    val_datagen = ImageDataGenerator(rescale=1./255)

    model = make_model()

    H = model.fit_generator(train_datagen.flow(train_x, train_y, batch_size=BS),
                  validation_data=val_datagen.flow(test_x, test_y, batch_size=BS), steps_per_epoch=len(train_x),
                  epochs=EPOCHS, verbose=1)

    model.save(MODEL_NAME)
    tb_callback = TensorBoard("./logs/" + model_name)

    plt.style.use("ggplot")
    plt.figure()
    N = EPOCHS
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on Cat/Dog")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig("plot")


def predict():
    x_test = np.load('test_images.npy')[0:10]

    model = load_model(MODEL_NAME)
    test_datagen = ImageDataGenerator(rescale=1.0/255)
    i=0
    text_labels = []

    plt.figure(figsize=(75, 75))

    for batch in test_datagen.flow(x_test, batch_size=1):
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

train_data = np.load('train_data.npy')
train_labels = np.load('train_labels.npy')

#train_model()

predict()
