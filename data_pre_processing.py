import keras
from keras.datasets import cifar10
import numpy as np
from scipy.misc import imresize
from scipy.misc import imshow
import pickle
import os
from sklearn.model_selection import train_test_split


def load_class_names():
	file_path = 'cifar-10-batches-py/batches.meta'
	with open(file_path, mode='rb') as file:
		meta = pickle.load(file, encoding='bytes')

	raw = meta[b'label_names']

	names = [x.decode('utf-8') for x in raw]

	return names

def getLabelIndexes(labels):
	if type(labels[0]) == str:
		ind = np.zeros((len(labels)),dtype=np.int)
		names = load_class_names()
		c = 0

		for i in range(len(names)):
			if names[i] in labels:
				ind[c] = i
				c += 1
		return ind
	elif type(labels[0]) == int:
		return labels
	else:
		raise ValueError("labels must be either string or index numbers")

def loadRawData():
	train,test = cifar10.load_data()
	return train,test

def loadTrainingData(train,labels,trainCases):
	sub_x_train = np.zeros((trainCases,32,32,3),dtype=np.int)
	sub_y_train = np.zeros((trainCases),dtype=np.int)

	x_train = train[0]
	y_train = train[1]

	ind = getLabelIndexes(labels)

	c = 0
	for i in range(len(y_train)):
		if y_train[i][0] in ind:
			sub_x_train[c] = x_train[i]
			sub_y_train[c] = y_train[i]
			c += 1
			if c >= trainCases:
				break
	return sub_x_train,sub_y_train

def loadTestingData(test,labels,testCases):
	sub_x_test = np.zeros((testCases,32,32,3),dtype=np.int)
	sub_y_test = np.zeros((testCases),dtype=np.int)

	x_test = test[0]
	y_test = test[1]

	ind = getLabelIndexes(labels)

	c = 0
	for i in range(len(y_test)):
		if y_test[i][0] in ind:
			sub_x_test[c] = x_test[i]
			sub_y_test[c] = y_test[i]
			c += 1
			if c >= testCases:
				break

	return sub_x_test,sub_y_test

def validationSplitter(x,y,proportion,shuffle):
	return train_test_split(x,y,proportion,shuffle=shuffle)

