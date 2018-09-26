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

augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
	augmented_images.append(image) #normal image
	augmented_measurements.append(measurement) #normal steer
	augmented_images.append(cv2.flip(image,1)) #flip image
	augmented_measurements.append(measurement*-1.0) #flip steer



X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Convolution2D,MaxPooling2D

#LeNet
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Convolution2D(6,5,5,activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(16,5,5,activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))
#it's regression problem, not classification 
#here we use mean square error loss function, no more softmax cross-entropy
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2,shuffle=True)

model.save('model.h5')

'''
Epoch 1/10
12857/12857 [==============================] - 27s 2ms/step - loss: 2.4406 - val_loss: 0.0143
Epoch 2/10
12857/12857 [==============================] - 26s 2ms/step - loss: 0.0124 - val_loss: 0.0128
Epoch 3/10
12857/12857 [==============================] - 26s 2ms/step - loss: 0.0107 - val_loss: 0.0118
Epoch 4/10
12857/12857 [==============================] - 26s 2ms/step - loss: 0.0095 - val_loss: 0.0114
Epoch 5/10
12857/12857 [==============================] - 26s 2ms/step - loss: 0.0086 - val_loss: 0.0113
Epoch 6/10
12857/12857 [==============================] - 26s 2ms/step - loss: 0.0078 - val_loss: 0.0113
Epoch 7/10
12857/12857 [==============================] - 26s 2ms/step - loss: 0.0071 - val_loss: 0.0116
Epoch 8/10
12857/12857 [==============================] - 26s 2ms/step - loss: 0.0065 - val_loss: 0.0116
Epoch 9/10
12857/12857 [==============================] - 27s 2ms/step - loss: 0.0058 - val_loss: 0.0122
Epoch 10/10
12857/12857 [==============================] - 26s 2ms/step - loss: 0.0051 - val_loss: 0.0133
'''