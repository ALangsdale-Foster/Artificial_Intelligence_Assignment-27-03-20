from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras import backend as K
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
print(tf.__version__)
import cv2
import glob
import pandas as pd
import os


def subtractOne(x):
    x = x - 1
    return x



##Format:
##    
##script root folder -> dataset folder -> train folder         -> images folder
##                                                                ->CSV file               ->images
##
##CSV Format:
##    Filename,             Category
##    example.jpg,        1
##    example2.jpg,      2


##initial config and reading csv file
train_data = pd.read_csv("dataset/train/train.csv")
print(train_data.head(5))
train_data["category"] = train_data["category"].apply(subtractOne) ##values should start at 0, not 1
print(train_data.head(5))
img_dims = (100,100,3)

data = []
labels = []

label_class = {
    0:"Cargo",
    1:"Military",
    2:"Carrier",
    3:"Cruise",
    4:"Tankers"}
##for testing purposes
i = 0
##create labels and load images
for index, row in train_data.iterrows():
    img_path = "dataset/train/images/" + row["image"]
    image = cv2.imread(img_path)
    image = cv2.resize(image, (img_dims[0], img_dims[1]))
    image = img_to_array(image)
    data.append(image)
    label = row["category"]
    labels.append([label])
    i += 1

##Preprocess data/labels
data = np.array(data, dtype = "float") / 255.0
labels = np.array(labels)

print(labels)

##Split the data into test/train sets
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size = 0.2,
                                                  random_state = 42)
trainY = to_categorical(trainY, num_classes = 5)
testY = to_categorical(testY, num_classes = 5)

##Augment dataset to improve performance
augment = ImageDataGenerator(rotation_range = 25,
                             width_shift_range = 0.1,
                             height_shift_range = 0.1,
                             shear_range = 0.2,
                             zoom_range = 0.2,
                             horizontal_flip = True,
                             fill_mode = "nearest")

##setup layers of model
model = keras.Sequential([keras.layers.Conv2D(32, (3,3), padding = "same", activation = "relu", input_shape = (100,100,3)),
                          keras.layers.MaxPooling2D(pool_size = (2,2)),
                          keras.layers.Conv2D(64, (3,3), padding = "same", activation = "relu"),
                          keras.layers.Dropout(0.3),
                          keras.layers.Conv2D(64, (3,3), padding = "same", activation = "relu"),
                          keras.layers.Dropout(0.3),
                          keras.layers.MaxPooling2D(pool_size = (2,2)),
                          keras.layers.Flatten(),
                          keras.layers.Dense(256, activation = "relu"),
                          keras.layers.Dense(256, activation = "relu"),
                          keras.layers.Dense(5, activation = "softmax")
                          ])

##compile the model
model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])

##apply the model to augmented dataset
model.fit_generator(augment.flow(trainX, trainY, batch_size = 64),
                                 validation_data = (testX, testY),
                                 steps_per_epoch = len(trainX) // 64,
                                 epochs = 10)



##testing to see if model works
img = data[i-1]
print(img.shape)
print("data-i")
print(img_path)

img = (np.expand_dims(img,0))
print(img.shape)

predictions_single = model.predict(img)

print(predictions_single)

cat = np.argmax(predictions_single[0])

print(label_class[cat])


##saving model to .json and weights to .h5
model_file = model.to_json()
with open("newModel.json", "w") as json_file:
    json_file.write(model_file)
model.save_weights("newWeights.h5")
