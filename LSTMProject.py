from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.model_selection import train_test_split

import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

def main():
	#seed=7
	#dataset = numpy.loadtxt('NFold/2015 to 2016.csv', delimiter=',', skiprows=1)
	#predset = numpy.loadtxt('NFold/2017 prediction test.csv', delimiter=',', skiprows=1)
	#outcsv = "NFold/2017 results.csv"
	dataset = numpy.loadtxt('2015 to 2016.csv', delimiter=',', skiprows=1)
	predset = numpy.loadtxt('2017 prediction test.csv', delimiter=',', skiprows=1)
	outcsv = "2017 results.csv"
	pX = predset[:, 0:8]
	pX = numpy.reshape(pX, (pX.shape[0], pX.shape[1], 1))
	pY = predset[:, 8]
	X = dataset[:, 0:8]
	X = numpy.reshape(X, (X.shape[0], X.shape[1], 1))
	Y = dataset[:,8]
	#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=seed)
	model=Sequential()
	model.add(LSTM(return_sequences=False, kernel_initializer='glorot_normal', input_shape=(None, 1), units=8))
	model.add(Dropout(0.2))
	model.add(Activation('relu'))
	model.add(Dense(units=1, kernel_initializer='glorot_normal'))
	model.add(Activation('sigmoid'))
	#model.add(Dense(units=1, kernel_initializer='glorot_normal'))
	#model.add(Activation('sigmoid'))
	model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
	model.fit(X, Y, epochs=5, batch_size=1, verbose=2)
	#model.fit(X_train, Y_train, epochs=30, batch_size=1, verbose=2)
	print(model.evaluate(pX, pY, batch_size=1, verbose=0, sample_weight=None))
	predicted=model.predict_classes(pX)
	print(predicted)
	numpy.savetxt(outcsv, predicted, delimiter=',')
	rmse=numpy.sqrt(((predicted-pY)**2).mean(axis=0))
	#for layer in model.layers:
	#	g=layer.get_config()
	#	h=layer.get_weights()
	#	print (g)
	#	print (h)

if __name__ == '__main__':
	main()