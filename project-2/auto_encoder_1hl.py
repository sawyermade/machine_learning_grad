'''
Daniel Sawyer u3363-7705
Tensorflow 2.1.0
Python 3.6.10

Usage: $ python3 auto_encoder_1hl.py
'''

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf, argparse, numpy as np

def parse_args():
	parser = argparse.ArgumentParser(description="Gets args")
	parser.add_argument(
		"--conv", 
		'-c',
		dest='conv',
		action="store_true", 
		help="Convolve instead of Dense"
	)
	parser.add_argument(
		"--learning-rate",
		'-lr',
		dest='learning_rate',
		type=float,
		default=0.6,
		help="Learning rate",
	)
	parser.add_argument(
		"--epochs",
		'-e',
		dest='epochs',
		type=int,
		default=1000,
		help="Number of epochs",
	)
	parser.add_argument(
		"--loss",
		'-l',
		dest='loss',
		type=str,
		default='mean_squared_error',
		help="Loss function",
	)
	parser.add_argument(
		"--activation-in",
		'-ai',
		dest='activation_in',
		type=str,
		default='tanh',
		help="Activation encoder",
	)
	parser.add_argument(
		"--activation-out",
		'-ao',
		dest='activation_out',
		type=str,
		default='tanh',
		help='Activation decoder',
	)
	return parser.parse_args()

def auto_encoder_model(x_train_shape, args):
	# Gets number of inputs
	num_inputs = 4
	num_hidden = 3

	# Creates model
	if args.conv:
		model = tf.keras.models.Sequential([
			tf.keras.layers.Conv1D(32, 3, padding='same', activation=args.activation_in, input_shape=x_train_shape),
			tf.keras.layers.MaxPooling1D(pool_size=2, strides=None),
			tf.keras.layers.Flatten(),
			tf.keras.layers.Dense(4, activation=args.activation_out)
		])

		# Compiles model
		model.compile(
			optimizer=tf.keras.optimizers.SGD(learning_rate=args.learning_rate),
			loss=args.loss,
			metrics=['accuracy']
		)
	
	else:
		model = tf.keras.models.Sequential([
			tf.keras.layers.Dense(num_hidden, activation=args.activation_in, input_shape=x_train_shape),
			tf.keras.layers.Dense(num_inputs, activation=args.activation_out)
		])

		# Compiles model
		model.compile(
			optimizer=tf.keras.optimizers.SGD(learning_rate=args.learning_rate),
			loss=args.loss,
			metrics=['accuracy']
		)

	return model

def main():
	# Gets args
	args = parse_args()

	# Creates input
	x_train = np.asarray([
		[1., 0, 0, 0],
		[0, 1., 0, 0],
		[0, 0, 1., 0],
		[0, 0, 0, 1.]
	])

	# Creates model
	model = auto_encoder_model(x_train.shape, args)

	# Fits model
	x_train = x_train.reshape(1, 4, 4)
	model.fit(x_train, x_train, epochs=args.epochs)

	# Predict
	pred = model.predict(x_train, verbose=2)
	if not args.conv:
		pred = pred.reshape(4, 4)

	# Prints predictions
	print(model.summary(), '\n')
	for p_row in pred:
		for p_val in p_row:
			print(f'{p_val:e}', end='  ')
		print()
	print()

if __name__ == '__main__':
	main()