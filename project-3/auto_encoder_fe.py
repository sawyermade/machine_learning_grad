'''
Daniel Sawyer u3363-7705
Tensorflow 2.1.0
Python 3.6.10

Usage: $ python3 auto_encoder_fe.py
'''

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf, argparse, numpy as np

def parse_args():
	parser = argparse.ArgumentParser(description="Gets args")
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
		default=300,
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

def auto_encoder_model_og(x_train_shape, args):
	# Gets number of inputs
	num_inputs = 4
	num_hidden = 3

	# Creates model
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
	x_train_shape = x_train.shape
	model = auto_encoder_model_og(x_train_shape, args)

	# Fits original model and extracts weights from layer 0
	x_train = x_train.reshape((1, 4, 4))
	model.fit(x_train, x_train, epochs=args.epochs)
	model_weights = model.layers[0].get_weights()

	# Load weights into temp model to get layer outputs for new_model inputs
	temp_model = tf.keras.Sequential([
		tf.keras.layers.Dense(3, activation='tanh', input_shape=x_train_shape)
	])
	temp_model.set_weights(model_weights)
	temp_model.compile(
		optimizer=tf.keras.optimizers.SGD(learning_rate=args.learning_rate), 
		loss=args.loss,
		metrics=['accuracy']
	)
	X = temp_model.predict(x_train)
	X_shape = X.shape[1:]

	# Trains new model using extracted features
	new_model = tf.keras.Sequential([
		tf.keras.layers.Dense(4, activation='tanh', input_shape=X_shape)
	])
	new_model.compile(
		optimizer=tf.keras.optimizers.SGD(learning_rate=args.learning_rate),
		loss=args.loss,
		metrics=['accuracy']
	)
	new_model.fit(X, x_train, epochs=args.epochs)
	
	# Gets new models prediction
	pred = new_model.predict(X)

	# Prints input and outputs of new model
	print(f'\nOutputs from original hidden layer, now input X:\n{X.reshape(X_shape)}\n')
	print(f'new_model prediction:\n{pred.reshape(x_train_shape)}\n')
	print(f'new_model prediction rounded:\n{pred.round().reshape(x_train_shape)}\n')
	
if __name__ == '__main__':
	main()