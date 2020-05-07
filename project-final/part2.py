import tensorflow as tf
from tensorflow.keras import layers, optimizers, models, datasets, utils
from tensorflow.keras.preprocessing.image import load_img, ImageDataGenerator
import numpy as np
from PIL import Image

import os, argparse, math, sys
import matplotlib.pyplot as plt

# Argument parser
def arg_parse(arg_list=None):
	parser = argparse.ArgumentParser(description="ML Final Project")

	# Batch size
	parser.add_argument(
		'--batch-size',
		'-bs',
		dest='batch_size',
		help='Batch size',
		type=int,
		default=128
	)

	# Number of Classes
	parser.add_argument(
		'--num-classes',
		'-c',
		dest='num_classes',
		help='Number of classes',
		type=int,
		default=5
	)

	# Number of Epochs
	parser.add_argument(
		'--epochs',
		'-e',
		dest='epochs',
		help='Number of Epochs',
		type=int,
		default=200
	)
	parser.add_argument(
		'--epochs-res',
		'-er',
		dest='epochs_res',
		help='Number of Epochs ResNet',
		type=int,
		default=650
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

	# ResNet Blocks
	parser.add_argument(
		'--resnet-blks',
		'-rb',
		dest='resnet_blks',
		help='Number of ResNet Blocks',
		type=int,
		default=32
	)

	# Learning Rate
	parser.add_argument(
		'--learning-rate',
		'-lr',
		dest='learning_rate',
		help='Learning Rate',
		type=float,
		default=0.001
	)
	parser.add_argument(
		'--learning-rate-res',
		'-lrr',
		dest='learning_rate_res',
		help='Learning Rate',
		type=float,
		default=0.001
	)

	# Learning Rate
	parser.add_argument(
		'--decay',
		'-de',
		dest='decay',
		help='Decay Rate',
		type=float,
		default=1e-10
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
		default='keras_cifar10_fcn_model'
	)

	# Data augmentation
	parser.add_argument(
		'--data-augmentation',
		'-da',
		dest='data_augmentation',
		action="store_false",
		help="Data Augmentation?"
	)

	# Batch Normalization
	parser.add_argument(
		'--batch-normalization',
		'-bn',
		dest='batch_norm',
		action="store_false",
		help="Batch Normalization?"
	)

	# Parses and returns args
	if arg_list:
		return parser.parse_args(args=arg_list)
	else:
		return parser.parse_args()

# Reorder classes
def class_reorder(y, class_list):
	y[0] = class_list.index(y[0])
	return y

# Data loader
def load_data5():
	# Loads train and test
	(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

	# Gets only 5 classes and reorders class numbers
	class_list = [2, 3, 4, 5, 7]
	x5_train = np.asarray([x for x, y in zip(x_train, y_train) if y[0] in class_list])
	x5_test = np.asarray([x for x, y in zip(x_test, y_test) if y[0] in class_list])
	y5_train = np.asarray([class_reorder(y, class_list) for y in y_train if y[0] in class_list])
	y5_test = np.asarray([class_reorder(y, class_list) for y in y_test if y[0] in class_list])

	# Returns data
	return (x5_train, y5_train), (x5_test, y5_test)

# FCN Horse test
def horse_test(model, class_labels, img_fname='images/horse_64.jpg'):
	img = np.asarray(load_img(img_fname)).astype('float32') / 255
	output_list = model.predict(np.expand_dims(img, axis=0)).tolist()[0]
	output = output_list.index(max(output_list))
	# print(f'output_list: {output_list}, output: {output}, label: {class_labels[output]}')
	return [class_labels[output], output_list[output]]

# Saves models
def save_model(args, model, model_name=None):
	if not model_name: model_name = args.model_name
	# Save model and weights
	if not os.path.isdir(args.save_dir):
		os.makedirs(args.save_dir)
	model_path = os.path.join(args.save_dir, model_name + '.h5')
	model.save(model_path)
	print(f'Saved trained model at {model_path}')

# Learning Rate Scheduler
def lr_sched(epoch):
	# lr = 1e-3
	# if epoch > 180: lr *= 0.5e-3
	# elif epoch > 160: lr *= 1e-3
	# elif epoch > 120: lr *= 1e-2
	# elif epoch > 80: lr *= 1e-1
	lr = 0.1
	# if epoch > 180: lr *= 0.5e-3
	# if epoch > 500: lr *= 0.0002 
	if epoch > 500: lr *= 0.0005 
	elif epoch > 300: lr *= 0.001
	elif epoch > 120: lr *= 0.01
	return lr

# Original Model but FCN
def og_model_fcn(args):
	# Model start
	model = models.Sequential()
	padding1 = 'same'
	padding2 = 'valid'
	
	# Fully convolutional input
	model.add(layers.Input(shape=(None, None, 3)))

	# First convolution block
	model.add(
		layers.Conv2D(filters=32, kernel_size=3, strides=1, padding=padding2)
	)
	model.add(layers.BatchNormalization())
	model.add(layers.Activation('relu'))	
	model.add(layers.Conv2D(32, (3, 3)))
	model.add(layers.BatchNormalization())
	model.add(layers.Activation('relu'))
	model.add(layers.MaxPooling2D(pool_size=(2, 2)))
	model.add(layers.Dropout(0.5))

	# Next conv block
	model.add(layers.Conv2D(64, (3, 3), padding=padding2))
	model.add(layers.BatchNormalization())
	model.add(layers.Activation('relu'))
	model.add(layers.Conv2D(64, (3, 3)))
	model.add(layers.BatchNormalization())
	model.add(layers.Activation('relu'))
	model.add(layers.MaxPooling2D(pool_size=(2, 2)))
	model.add(layers.Dropout(0.5))

	# Fully convolutional output
	model.add(layers.Conv2D(filters=256, kernel_size=5, strides=1, padding=padding2))
	model.add(layers.Dropout(0.5))
	model.add(layers.BatchNormalization())
	model.add(layers.Activation('relu'))
	model.add(layers.Conv2D(args.num_classes, 1, 1, padding2))
	model.add(layers.GlobalMaxPooling2D())
	model.add(layers.Activation('softmax'))

	# Initiate optimizer
	opt = optimizers.Adamax(lr=args.learning_rate, decay=args.decay)

	# Let's train the model using RMSprop
	model.compile(
		loss='categorical_crossentropy',
		optimizer=opt,
		metrics=['accuracy']
	)

	# Returns model
	return model

# ResNet Block Shallow
def resnet_block_shallow(input_layer, filters, ksize):
	l = layers.Conv2D(filters, ksize, 1, 'same')(input_layer)
	l = layers.BatchNormalization()(l)
	l = layers.Activation('relu')(l)

	l = layers.Conv2D(filters, ksize, 1, 'same')(l)
	l = layers.BatchNormalization()(l)

	l = layers.Add()([l, input_layer])
	l = layers.Activation('relu')(l)
	return l

# ResNet Block Deep
def resnet_block_deep(input_layer, filters, ksize):
	l = layers.Conv2D(filters, 1, 1, 'same')(input_layer)
	l = layers.BatchNormalization()(l)
	l = layers.Activation('relu')(l)

	l = layers.Conv2D(filters, ksize, 1, 'same')(l)
	l = layers.BatchNormalization()(l)
	l = layers.Activation('relu')(l)

	l = layers.Conv2D(filters, 1, 1, 'same')(l)
	l = layers.Add()([l, input_layer])
	l = layers.Activation('relu')(l)
	# l = layers.Dropout(0.2)(l)
	return l

# Resnet FCN
def resnet_fcn(args):
	# Input shape
	input_shape = (None, None, 3)
	# input_shape = (32, 32, 3)

	# ResNet Blocks
	resent_blks = args.resnet_blks

	# Model Input layers
	input_layer = layers.Input(shape=input_shape)
	l = layers.Conv2D(32, 3)(input_layer)
	l = layers.BatchNormalization()(l)
	l = layers.Activation('relu')(l)
	l = layers.Conv2D(64, 3)(l)
	l = layers.BatchNormalization()(l)
	l = layers.Activation('relu')(l)
	# l = layers.MaxPooling2D()(l)
	l = layers.AveragePooling2D()(l)
	l = layers.Dropout(0.3)(l)

	# ResNet Blocks
	for i in range(resent_blks):
		if resent_blks <= 10:
			l = resnet_block_shallow(l, 64, 3)
		else:
			l = resnet_block_deep(l, 64, 3)
	l = layers.Dropout(0.5)(l)

	# Final Convolutions
	l = layers.Conv2D(64, 3)(l)
	l = layers.BatchNormalization()(l)
	l = layers.Activation('relu')(l)
	# l = layers.GlobalAveragePooling2D()(l)
	# l = layers.GlobalMaxPooling2D()(l)
	# l = layers.MaxPooling2D()(l)
	l = layers.AveragePooling2D()(l)
	l = layers.Dropout(0.5)(l)

	# Fully convolutional output
	l = layers.Conv2D(filters=512, kernel_size=6, strides=1)(l)
	l = layers.BatchNormalization()(l)
	# l = layers.Dropout(0.5)(l)
	l = layers.Activation('relu')(l)
	l = layers.Conv2D(args.num_classes, 1, 1)(l)
	# l = layers.GlobalMaxPooling2D()(l)
	l = layers.GlobalAveragePooling2D()(l)
	output_layer = layers.Activation('softmax')(l)

	# Final model
	model = tf.keras.Model(input_layer, output_layer)

	# Initiate optimizer
	# opt = optimizers.Adam(learning_rate=args.learning_rate_res)
	# opt = optimizers.Adamax(learning_rate=args.learning_rate_res)
	opt = optimizers.Adamax(learning_rate=lr_sched(0))
	

	# Let's train the model using RMSprop
	model.compile(
		loss='categorical_crossentropy',
		optimizer=opt,
		metrics=['accuracy']
	)

	return model

def main():
	# Args
	args = arg_parse()
	if not os.path.exists(args.save_dir):
		os.makedirs(args.save_dir)

	# Class labels
	class_labels = ['bird', 'cat', 'deer', 'dog', 'horse']

	# Set CUDA Card
	physical_devices = tf.config.experimental.list_physical_devices('GPU')
	tf.config.experimental.set_memory_growth(physical_devices[0], True)
	# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
	# os.environ["CUDA_VISIBLE_DEVICES"]=args.cuda

	# Load data
	(x_train, y_train), (x_test, y_test) = load_data5()

	# Convert class vectors to binary class matrices.
	y_train = utils.to_categorical(y_train, args.num_classes)
	y_test = utils.to_categorical(y_test, args.num_classes)

	# Normalize
	x_train = x_train.astype('float32') / 255
	x_test = x_test.astype('float32') / 255

	# Original Model with FCN mods from part1
	# model_og_fcn_name = 'keras_cifar10_og_fcn_model'
	# model_og_fcn = og_model_fcn(args)
	# Callbacks
	# checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='saved_models/keras_cifar10_og_fcn_model-checkpoint.h5', monitor='val_accuracy', verbose=1, save_best_only=True)
	# lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_sched)
	# lr_red = tf.keras.callbacks.ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
	# callbacks = [checkpoint, lr_red, lr_scheduler]
	# model_og_fcn.fit(
	# 	x_train, 
	# 	y_train,
	# 	batch_size=args.batch_size,
	# 	epochs=args.epochs,
	# 	validation_data=(x_test, y_test),
	# 	shuffle=True,
	#	callbacks=True
	# )

	# FCN ResNet Model
	model_resnet_fcn_name = 'keras_cifar10_resnet_fcn_model' 
	model_resnet_fcn = resnet_fcn(args)
	
	# Callbacks
	checkpoint = tf.keras.callbacks.ModelCheckpoint(
		filepath='saved_models/keras_cifar10_resnet_fcn_model-checkpoint.h5', 
		monitor='val_accuracy', 
		verbose=1, 
		save_best_only=True
	)
	lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_sched)
	lr_red = tf.keras.callbacks.ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
	callbacks = [checkpoint, lr_red, lr_scheduler]

	if not args.data_augmentation:
		model_resnet_fcn.fit(
			x_train, 
			y_train,
			batch_size=args.batch_size,
			epochs=args.epochs_res,
			validation_data=(x_test, y_test),
			shuffle=True,
			callbacks=callbacks
		)
	else:
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
		model_resnet_fcn.fit_generator(
			datagen.flow(
				x_train, 
				y_train,
				batch_size = args.batch_size
			),
			steps_per_epoch = x_train.shape[0] // args.batch_size,
			epochs = args.epochs_res,
			validation_data = (x_test, y_test),
			callbacks = callbacks
		)

	# Save models/weights
	# save_model(args, model_og_fcn, model_og_fcn_name)
	# save_model(args, model_resnet_fcn, model_resnet_fcn_name)

	# Get Scores for models
	# scores_og_fcn = model_og_fcn.evaluate(x_test, y_test, verbose=1)
	model_resnet_fcn = tf.keras.models.load_model('saved_models/keras_cifar10_resnet_fcn_model-checkpoint.h5')
	scores_resnet_fcn = model_resnet_fcn.evaluate(x_test, y_test, verbose=1)

	# Get horse test for FCN models to show FCN works
	# label_og_fcn, conf_og_fcn = horse_test(model_og_fcn, class_labels)
	label_resnet_fcn, conf_resnet_fcn = horse_test(model_resnet_fcn, class_labels)
	
	# Other FCN tests
	gt_list = ['bird', 'cat', 'cat', 'deer', 'dog', 'dog', 'horse', 'horse']
	out_list = []
	out_list.append(horse_test(model_resnet_fcn, class_labels, 'images/bird_64.jpg'))
	out_list.append(horse_test(model_resnet_fcn, class_labels, 'images/cat_64.jpg'))
	out_list.append(horse_test(model_resnet_fcn, class_labels, 'images/cat2_64.jpg'))
	out_list.append(horse_test(model_resnet_fcn, class_labels, 'images/deer_64.jpg'))
	out_list.append(horse_test(model_resnet_fcn, class_labels, 'images/dog_64.jpg'))
	out_list.append(horse_test(model_resnet_fcn, class_labels, 'images/dog2_64.jpg'))
	out_list.append(horse_test(model_resnet_fcn, class_labels, 'images/horse_64.jpg'))
	out_list.append(horse_test(model_resnet_fcn, class_labels, 'images/horse2_64.jpg'))

	# Save scores for models and horse test
	# results_og_fcn = f'\nOriginal FCN:\nTest loss: {scores_og_fcn[0]}, Test accuracy: {scores_og_fcn[1]}\nHorse Test not 32x32x3 NO RESIZE: {label_og_fcn} @ {conf_og_fcn}%\n'
	results_resnet_fcn = (
		f'\nResNet FCN:'
		f'\nTest loss: {scores_resnet_fcn[0]}, Test accuracy: {scores_resnet_fcn[1]}'
		f'\nHorse Test not 32x32x3 NO RESIZE: {label_resnet_fcn} @ {conf_resnet_fcn}%\n'
	)
	# print(results_og_fcn)
	print(results_resnet_fcn)
	with open('output_part2.txt', 'w') as of:
		# Write model summaries
		# of.write('ORIGINAL FCN MODEL SUMMARY:\n')
		# model_og_fcn.summary(print_fn=lambda x: of.write(x + '\n'))
		# of.write('\n')
		of.write('RESNET FCN MODEL SUMMARY:\n')
		model_resnet_fcn.summary(print_fn=lambda x: of.write(x + '\n'))
		of.write('\n')

		# Write results
		# of.write(results_og_fcn)
		of.write(results_resnet_fcn)

		# Prints FCN tests
		print('\n')
		of.write('\n')
		for o, gt in zip(out_list, gt_list):
			label, conf = o
			fcn_test_str = f'{gt}: {label} @ {conf}%'
			print(f'{gt}: {label} @ {conf}%')
			of.write(fcn_test_str + '\n')

if __name__ == '__main__':
	main()