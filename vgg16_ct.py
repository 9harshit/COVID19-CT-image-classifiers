#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 12:54:15 2020

@author: harshit
"""




# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages

import tensorflow as tf

# Initialising the CNN
classifier_base = tf.keras.applications.VGG16(input_shape = (64,64,3),
    include_top=False
)

# Compiling the CNN
classifier_base.summary()
classifier_base.trainable = True

golab_avg_layer = tf.keras.layers.GlobalAveragePooling2D()(classifier_base.output)
prediction_layer = tf.keras.layers.Dense(units = 1 , activation= "sigmoid")(golab_avg_layer)

classifier = tf.keras.models.Model(inputs = classifier_base.input, outputs = prediction_layer)
classifier.summary()
# Convolutional Neural Network
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')
from keras.callbacks import  EarlyStopping

early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=20, verbose=1, mode='auto')

classifier.fit_generator(training_set,
                         epochs = 5,
                         validation_data = test_set,
                         callbacks = early)


classifier.save("vgg166.h5")

import matplotlib.pyplot as plt
plt.plot(classifier.history.history['accuracy'])
plt.plot(classifier.history.history['val_accuracy'])
plt.plot(classifier.history.history['loss'])
plt.plot(classifier.history.history['val_loss'])
plt.title("model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Accuracy","Validation Accuracy","loss","Validation Loss"])
plt.show()