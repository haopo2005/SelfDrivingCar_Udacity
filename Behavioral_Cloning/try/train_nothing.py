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
from keras.layers import Flatten, Dense

model = Sequential()
model.add(Flatten(input_shape=(160,320,3)))
model.add(Dense(1))

#it's regression problem, not classification 
#here we use mean square error loss function, no more softmax cross-entropy
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2,shuffle=True)

model.save('model.h5')


'''
Epoch 1/10
6428/6428 [==============================] - 8s 1ms/step - loss: 8248721.5205 - val_loss: 7114.7119
Epoch 2/10
6428/6428 [==============================] - 3s 540us/step - loss: 3859.2249 - val_loss: 2607.8044
Epoch 3/10
6428/6428 [==============================] - 3s 531us/step - loss: 2727.5907 - val_loss: 2177.9355
Epoch 4/10
6428/6428 [==============================] - 3s 544us/step - loss: 2449.8551 - val_loss: 1789.2704
Epoch 5/10
6428/6428 [==============================] - 4s 548us/step - loss: 1823.3589 - val_loss: 1508.8458
Epoch 6/10
6428/6428 [==============================] - 4s 549us/step - loss: 1916.3883 - val_loss: 1866.3350
Epoch 7/10
6428/6428 [==============================] - 4s 557us/step - loss: 1469.6970 - val_loss: 1496.8223
Epoch 8/10
6428/6428 [==============================] - 3s 538us/step - loss: 2270.9337 - val_loss: 1387.5320
Epoch 9/10
6428/6428 [==============================] - 4s 546us/step - loss: 3975.1205 - val_loss: 3076.7158
Epoch 10/10
6428/6428 [==============================] - 3s 542us/step - loss: 155481.7300 - val_loss: 3768733.5821

'''