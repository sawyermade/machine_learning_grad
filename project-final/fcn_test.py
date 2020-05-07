import tensorflow as tf
from tensorflow.keras import layers, optimizers, models, datasets, utils
from tensorflow.keras.preprocessing.image import load_img
import numpy as np

from part2 import horse_test

import os, argparse, math, sys

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def main():
	class_labels = ['bird', 'cat', 'deer', 'dog', 'horse']
	gt_list = ['bird', 'cat', 'cat', 'deer', 'dog', 'dog', 'horse', 'horse']
	model_path = sys.argv[1]
	model = models.load_model(model_path)
	out_list = []
	out_list.append(horse_test(model, class_labels, 'images/bird_64.jpg'))
	out_list.append(horse_test(model, class_labels, 'images/cat_64.jpg'))
	out_list.append(horse_test(model, class_labels, 'images/cat2_64.jpg'))
	out_list.append(horse_test(model, class_labels, 'images/deer_64.jpg'))
	out_list.append(horse_test(model, class_labels, 'images/dog_64.jpg'))
	out_list.append(horse_test(model, class_labels, 'images/dog2_64.jpg'))
	out_list.append(horse_test(model, class_labels, 'images/horse_64.jpg'))
	out_list.append(horse_test(model, class_labels, 'images/horse2_64.jpg'))
	
	print('\n')
	for o, gt in zip(out_list, gt_list):
		label, conf = o
		print(f'{gt}: {label} @ {conf}%')

	# # model.summary()
	# img = np.asarray(load_img(img_fname))
	# output_list = model.predict(np.expand_dims(img, axis=0)).tolist()[0]
	# print(f'output_list = {output_list}')
	# output = output_list.index(max(output_list))
	
	# print(f'Prediction: {class_labels[output]} @ {output_list[output]}% confidence')

if __name__ == '__main__':
	main()