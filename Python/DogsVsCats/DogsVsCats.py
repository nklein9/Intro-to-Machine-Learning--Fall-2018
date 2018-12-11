# Nicholas Klein
# Python 3.6
# Created 11/20/18, Last edit 12/4/18
# CNN- Dogs Vs Cats


# from __future__ import print_function
import tensorflow
import cv2                 # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import os                  # dealing with directories
import matplotlib.pylab as plt
from random import shuffle # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import tqdm      # a nice pretty percentage bar for tasks.
from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.models import Sequential


TRAIN_DIR = r'C:\Users\nklei\Documents\Homework\ML DogsVsCatsData\train'
TEST_DIR = r'C:\Users\nklei\Documents\Homework\ML DogsVsCatsData\test'
IMG_SIZE = 100
batch_size = 128
num_classes = 2
epochs = 10
LR = 1e-3

MODEL_NAME = 'dogsvscats-{}-{}.model'.format(LR, '2conv-basic')


def label_img(img):
    word_label = img.split('.')[-3]
    # conversion to one-hot array [cat,dog]
    #                            [much cat, no dog]
    if word_label == 'cat': return [1,0]
    #                             [no cat, very doggo]
    elif word_label == 'dog': return [0,1]


def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR,img)
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        training_data.append([np.array(img),np.array(label)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data


def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)
        img_num = img.split('.')[0]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        testing_data.append([np.array(img), img_num])

    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data


train_data = create_train_data()
# If you have already created the dataset:
# train_data = np.load('train_data.npy')

input_shape = [IMG_SIZE, IMG_SIZE, 1]

train = train_data[:-7500]
test = train_data[-7500:]


x_train = np.array([i[0] for i in train])
# x_train = np.array(train[:, 0])
print('x_train: ', x_train.shape)
x_train = x_train.reshape(x_train.shape[0], IMG_SIZE, IMG_SIZE, 1)
# y_train = train[:, 1]
y_train = [i[1] for i in train]
# print('y_train: ', y_train.shape)

x_test = np.array([i[0] for i in test])
# x_test = np.array(test[:, 0])
x_test = x_test.reshape(x_test.shape[0], IMG_SIZE, IMG_SIZE, 1)
y_test = [i[1] for i in test]
# y_test = test[:, 1]

# convert the data to the right type
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices - this is for use in the
# categorical_crossentropy loss below
# y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes)
y_train = np.array(y_train)
# y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes)
y_test = np.array(y_test)
# print('y_train 2: ', y_train.shape)

model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.summary()

model.compile(loss=tensorflow.keras.losses.binary_crossentropy,
              optimizer=tensorflow.keras.optimizers.Adam(),
              metrics=['accuracy'])


class AccuracyHistory(tensorflow.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))


history = AccuracyHistory()


model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[history])
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
plt.plot(range(1, 11), history.acc)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

"""
import cv2                 # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import os                  # dealing with directories
import tensorflow
import matplotlib.pyplot as plt
from random import shuffle # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import tqdm      # a nice pretty percentage bar for tasks.
from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.models import Sequential
"""
# TRAIN_DIR = r'C:\Users\nklei\Documents\Homework\ML DogsVsCatsData\train'
#TEST_DIR = r'C:\Users\nklei\Documents\Homework\ML DogsVsCatsData\test'
#IMG_SIZE = 250
#LR = 1e-3
#batch_size = 128
"""
MODEL_NAME = 'dogsvscats-{}-{}.model'.format(LR, '2conv-basic')


def label_img(img):
    word_label = img.split('.')[-3]
    # conversion to one-hot array [cat,dog]
    #                            [much cat, no dog]
    if word_label == 'cat': return [1,0]
    #                             [no cat, very doggo]
    elif word_label == 'dog': return [0,1]


def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR,img)
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        training_data.append([np.array(img),np.array(label)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data


def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)
        img_num = img.split('.')[0]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        testing_data.append([np.array(img), img_num])

    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data


# train_data = create_train_data()
# If you have already created the dataset:
train_data = np.load('train_data.npy')

input_shape = [IMG_SIZE, IMG_SIZE, 1]
# create model
model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(loss=tensorflow.keras.losses.categorical_crossentropy,
              optimizer=tensorflow.keras.optimizers.SGD(lr=0.01),
              metrics=['accuracy'])


if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded!')


class AccuracyHistory(tensorflow.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))


history = AccuracyHistory()

train = train_data[:-500]
test = train_data[-500:]


X = np.array([i[0] for i in train])
X.reshape(X.shape[0], IMG_SIZE, IMG_SIZE, 1)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test])
test_x.reshape(test_x.shape[0], IMG_SIZE, IMG_SIZE, 1)
test_y = [i[1] for i in test]


model.fit(X, Y,
          batch_size=batch_size,
          epochs=10,
          verbose=1,
          validation_data=(test_x, test_y),
          callbacks=[history])


score = model.evaluate(test_x, test_y, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

plt.plot(range(1,11), history.acc)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()
"""
