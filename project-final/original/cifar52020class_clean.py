#!/usr/bin/env python
# coding: utf-8

# It's not the future!
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

''' 
Tensorflow/keras code modified by L.O. Hall 
4/13/20 to load 5 of 6 animal classes for training 
(no frog)
'''
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import cifar10
from tensorflow.python.util.tf_export import tf_export

import os, math, numpy as np, argparse, sys

# Argument parser
def arg_parse(arg_list=None):
	parser = argparse.ArgumentParser(description="Detectron2 Training")

	# Batch size
	parser.add_argument(
		'--batch-size',
		'-bs',
		dest='batch_size',
		help='Batch size',
		type=int,
		default=32
	)

	# Number of Classes
	parser.add_argument(
		'--num-classes',
		'-c',
		dest='num_classes',
		help='Number of classes',
		type=int,
		default=10
	)

	# Number of Epochs
	parser.add_argument(
		'--epochs',
		'-e',
		dest='epochs',
		help='Number of Epochs',
		type=int,
		default=100
	)

	# Number of Predictions
	parser.add_argument(
		'--num-predictions',
		'-np',
		dest='num_predictions',
		help='Number of Predictions',
		type=int,
		default=20
	)

	# Cuda Device Visible
	parser.add_argument(
		'--cuda',
		'-cu',
		dest='cuda',
		help='CUDA card to use',
		type=str,
		default='0'
	)

	# Save Directory
	parser.add_argument(
		'--out-dir',
		'-od',
		dest='save_dir',
		help='Output Directory Path',
		type=str,
		default='saved_models'
	)

	# Save Filename
	parser.add_argument(
		'--output',
		'-o',
		dest='model_name',
		help='Output Filename',
		type=str,
		default='keras_cifar10_trained_model.h5'
	)

	# Data augmentation
	parser.add_argument(
		'--data-augmentation',
		'-da',
		dest='data_augmentation',
		action="store_true",
		help="Data Augmentation?"
	)

	# Parses and returns args
	if arg_list:
		return parser.parse_args(args=arg_list)
	else:
		return parser.parse_args()

"""
Loads CIFAR10 dataset. However, just 5 classes, all animals except frog
Returns:
Tuple of Numpy arrays: (x_train, y_train), (x_test, y_test)
"""
def load_data5():
	# Loads train and test
	'''
	dirname = 'cifar-10-batches-py'
	origin = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
	path = get_file(dirname, origin=origin,  untar=True)
	path= './cifar-10-batches-py'
	'''
	(x_train, y_train), (x_test, y_test) = cifar10.load_data()

	# Below shows a test class has 1000 examples
	tclasscount = np.zeros((10,), dtype=int)
	for i in range(len(y_test)):
		tclasscount[y_test[i, 0]] += 1
	print(f'Test class count: {tclasscount}')

	# Number of samples etc
	num_train_samples = 50000
	num_5_class = 25000
	num_5_test = 5000
	print(f'x_train shape orig: {x_train.shape}')
	print(f'More: {x_train.shape[1:]}')
	print(f'y_test shape: {y_test.shape}')

	# Gets only 5 classes
	class_list = [2, 3, 4, 5, 7]
	x5_train = np.asarray([x for x, y in zip(x_train, y_train) if y[0] in class_list])
	x5_test = np.asarray([x for x, y in zip(x_test, y_test) if y[0] in class_list])
	y5_train = np.asarray([y for y in y_train if y[0] in class_list])
	y5_test = np.asarray([y for y in y_test if y[0] in class_list])

	# Prints 5 classes info
	c2, c3, c4, c5, c7 = [len(y5_test[y5_test[:, 0] == c]) for c in class_list]
	print(f'y5_test.shape: {y5_test.shape}')
	print(f'c2count: {c2}, c3count: {c3}, c4count: {c4}, c5count:{c5}, c7count:{c7}')
	print('Data Loaded\n')

	# Returns data
	return x5_train, y5_train, x5_test, y5_test

# Main
def main():
	# Args
	args = arg_parse()
	batch_size = args.batch_size
	num_classes = args.num_classes
	epochs = args.epochs 
	data_augmentation = args.data_augmentation
	num_predictions = args.num_predictions
	save_dir = args.save_dir
	model_name = args.model_name

	# Set CUDA Card
	physical_devices = tf.config.experimental.list_physical_devices('GPU')
	tf.config.experimental.set_memory_growth(physical_devices[0], True)
	# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
	# os.environ["CUDA_VISIBLE_DEVICES"]=args.cuda

	# Loads data
	x_train, y_train, x_test, y_test = load_data5()
	print(f'x_train shape: {x_train.shape}')
	print(f'train samples: {x_train.shape[0]}')
	print(f'test samples: {x_test.shape[0]}')

	# Model
	steps_for_epoch = math.ceil(x_train.shape[0] / batch_size)
	print(f'num_classes: {num_classes}')
	print(f'y_train',y_train)
	print('y_test',y_test)
	# Convert class vectors to binary class matrices.
	y_train = to_categorical(y_train, num_classes)
	y_test = to_categorical(y_test, num_classes)

	model = Sequential()
	model.add(
		Conv2D(32, (3, 3), padding='same',
		input_shape=x_train.shape[1:])
	)
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
	model.compile(
		loss='categorical_crossentropy',
		optimizer=opt,
		metrics=['accuracy']
	)

	# Normalize
	x_train = x_train.astype('float32') / 255
	x_test = x_test.astype('float32') / 255

	# No augment
	if not data_augmentation:
		print('Not using data augmentation.')
		model.fit(
			x_train, 
			y_train,
			batch_size=batch_size,
			epochs=epochs,
			validation_data=(x_test, y_test),
			shuffle=True
		)

	# With augments
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
			validation_split=0.0
		)

		# Compute quantities required for feature-wise normalization
		# (std, mean, and principal components if ZCA whitening is applied).
		datagen.fit(x_train)

		# Fit the model on the batches generated by datagen.flow().
		model.fit_generator(
			datagen.flow(x_train, y_train,
			batch_size=batch_size),
			steps_per_epoch = steps_for_epoch,
			epochs=epochs,
			validation_data=(x_test, y_test),
			workers=4
		)

	# Save model and weights
	if not os.path.isdir(save_dir):
		os.makedirs(save_dir)
	model_path = os.path.join(save_dir, model_name)
	model.save(model_path)
	print(f'Saved trained model at {model_path}')

	# Score trained model.
	scores = model.evaluate(x_test, y_test, verbose=1)
	print(f'Test loss: {scores[0]}')
	print(f'Test accuracy: {scores[1]}')

if __name__ == '__main__':
	main()