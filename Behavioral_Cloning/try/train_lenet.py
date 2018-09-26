import csv
import cv2

lines = []
with open('../data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

images = []
measurements = []
for line in lines[1:]:
	source_path = line[0] #only center camera
	filename = source_path.split('/')[-1]
	current_path = '../data/IMG/' + filename
	image = cv2.imread(current_path)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	images.append(image)
	measurement = float(line[3]) #only steering
	measurements.append(measurement)
	
import numpy as np

X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers import Convolution2D,MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Convolution2D(6,5,5,activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(.2))
model.add(Convolution2D(16,5,5,activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(.2))
model.add(Flatten())
model.add(Dense(120))
model.add(Dropout(.4))
model.add(Dense(84))
model.add(Dropout(.4))
model.add(Dense(1))
#it's regression problem, not classification 
#here we use mean square error loss function, no more softmax cross-entropy
model.compile(loss='mse', optimizer='adam')
model.summary()
model.fit(X_train, y_train, validation_split=0.2,shuffle=True)

model.save('model.h5')

'''
Epoch 1/10
6428/6428 [==============================] - 14s 2ms/step - loss: 7.4717 - val_loss: 0.0172
Epoch 2/10
6428/6428 [==============================] - 13s 2ms/step - loss: 0.0138 - val_loss: 0.0148
Epoch 3/10
6428/6428 [==============================] - 13s 2ms/step - loss: 0.0122 - val_loss: 0.0133
Epoch 4/10
6428/6428 [==============================] - 13s 2ms/step - loss: 0.0111 - val_loss: 0.0125
Epoch 5/10
6428/6428 [==============================] - 13s 2ms/step - loss: 0.0103 - val_loss: 0.0117
Epoch 6/10
6428/6428 [==============================] - 13s 2ms/step - loss: 0.0098 - val_loss: 0.0115
Epoch 7/10
6428/6428 [==============================] - 13s 2ms/step - loss: 0.0093 - val_loss: 0.0112
Epoch 8/10
6428/6428 [==============================] - 13s 2ms/step - loss: 0.0089 - val_loss: 0.0112
Epoch 9/10
6428/6428 [==============================] - 13s 2ms/step - loss: 0.0086 - val_loss: 0.0113
Epoch 10/10
6428/6428 [==============================] - 13s 2ms/step - loss: 0.0082 - val_loss: 0.0108
'''