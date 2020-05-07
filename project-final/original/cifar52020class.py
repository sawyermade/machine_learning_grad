#!/usr/bin/env python
# coding: utf-8


from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# Tensorflow/keras code modified by L.O. Hall 4/13/20 to load 5 of 6 animal classes for training (no frog)

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical

import os
import math

import numpy as np

from tensorflow.keras import backend as K
from tensorflow.keras.datasets import cifar10
from tensorflow.python.util.tf_export import tf_export

batch_size = 32
num_classes = 10
epochs = 100
#data_augmentation = True
data_augmentation = False
#num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'


def load_data5():
  """Loads CIFAR10 dataset. However, just 5 classes, all animals except frog
  Returns:
      Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
  """
#  dirname = 'cifar-10-batches-py'
#  origin = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
#  path = get_file(dirname, origin=origin,  untar=True)
#  path= './cifar-10-batches-py'
  (x_train, y_train), (x_test, y_test) = cifar10.load_data()
# Below shows a test class has 999 examples instead of the claimed 1000
#  tclasscount=np.zeros((10,), dtype=int)
#  for i in range(0, len(y_test)-1):
#    tclasscount[y_test[i][0]]= tclasscount[y_test[i][0]] + 1
#  print('Test class count',tclasscount)
  num_train_samples = 50000
  num_5_class = 25000
  num_5_test = 4999 # should be 5000 if all the categories had 1000 in them but they do not. One is missing.
  print('x_train shape orig:', x_train.shape)
  print('More:', x_train.shape[1:])
  print('y_test shape',y_test.shape)

  x5_train = np.empty((num_5_class, 32, 32, 3), dtype='uint8')
  y5_train = np.empty((num_5_class,), dtype='uint8')

  count=0

  for i in range(0, len(y_train)-1):
   if (y_train[i][0] == 2) or (y_train[i][0] == 3) or (y_train[i][0] == 4) or (y_train[i][0] == 5) or (y_train[i][0] == 7):
    x5_train[count]=x_train[i]
    y5_train[count]=y_train[i]
    count=count+1
   
    # find test data of interest
  count=0
  x5_test=np.empty((num_5_test, 32, 32, 3), dtype='uint8')
  y5_test= np.empty((num_5_test,), dtype='uint8')

  for i in range(0, len(y_test)-1):
   if (y_test[i][0] == 2) or (y_test[i][0] == 3) or (y_test[i][0] == 4) or (y_test[i][0] == 5) or (y_test[i][0] == 7):
    x5_test[count]=x_test[i]
    y5_test[count]=y_test[i]
    count=count+1
# Below shows class 7 is only 999 and not 1000 examples!!!  One horse got away it seems.
#    if(y_test[i][0] == 2):
#     c2=c2+1
#    if(y_test[i][0] == 3):
#     c3=c3+1
#    if(y_test[i][0] == 4):
#     c4=c4+1
#    if(y_test[i][0] == 5):
#     c5=c5+1
#    if(y_test[i][0] == 7):
#     c7=c7+1
#  print('c2count, c3count, c4count, c5count, c7count',c2,c3,c3,c5,c7)
#  print('y5tstshape',y5_test.shape, count)
#  print('y5tst',y5_test)
#  return (x_train, y_train), (x_test, y_test)
  return (x5_train, y5_train), (x5_test, y5_test)



(x_train, y_train), (x_test, y_test) = load_data5()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

steps_for_epoch = math.ceil(x_train.shape[0] / batch_size)
print('num classes',num_classes)
print('y_train',y_train)
print('y_test',y_test)
# Convert class vectors to binary class matrices.
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# fully connected layer of 512 coming
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
opt = tf.keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)
else:   # THIS CODE NOT TESTED IN TENSORFLOW 2.0.  IT IS AS IS!!!
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
        shear_range=0.,  # set range for random shear
        zoom_range=0.,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                        steps_per_epoch = steps_for_epoch,
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        workers=4)

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])



