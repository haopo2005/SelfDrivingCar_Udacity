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
	images.append(image)
	measurement = float(line[3]) #only steering
	measurements.append(measurement)
	
import numpy as np

X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Flatten())
model.add(Dense(1))

#it's regression problem, not classification 
#here we use mean square error loss function, no more softmax cross-entropy
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2,shuffle=True)

model.save('model.h5')

'''
Epoch 1/10
6428/6428 [==============================] - 4s 623us/step - loss: 1.4921 - val_loss: 1.6910
Epoch 2/10
6428/6428 [==============================] - 4s 573us/step - loss: 3.7729 - val_loss: 2.2169
Epoch 3/10
6428/6428 [==============================] - 4s 577us/step - loss: 4.2946 - val_loss: 2.7926
Epoch 4/10
6428/6428 [==============================] - 4s 582us/step - loss: 3.4507 - val_loss: 2.1698
Epoch 5/10
6428/6428 [==============================] - 4s 582us/step - loss: 3.4933 - val_loss: 4.3008
Epoch 6/10
6428/6428 [==============================] - 4s 584us/step - loss: 3.5779 - val_loss: 1.6661
Epoch 7/10
6428/6428 [==============================] - 4s 593us/step - loss: 4.6673 - val_loss: 2.0734
Epoch 8/10
6428/6428 [==============================] - 4s 565us/step - loss: 3.5077 - val_loss: 2.7254
Epoch 9/10
6428/6428 [==============================] - 4s 566us/step - loss: 3.1123 - val_loss: 2.2634
Epoch 10/10
6428/6428 [==============================] - 4s 576us/step - loss: 4.7257 - val_loss: 9.9618

'''