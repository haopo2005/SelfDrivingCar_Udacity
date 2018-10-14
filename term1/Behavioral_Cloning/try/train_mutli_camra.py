import csv
import os

#use generator yield keyword, more memory efficient
'''
Generators can be a great way to work with large amounts of data. Instead of storing the preprocessed data in memory all at once, 
using a generator you can pull pieces of the data and process them on the fly only when you need them, which is much more memory-efficient.
'''
lines = []
with open('../data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines[1:], test_size=0.2)

import cv2
import numpy as np
import sklearn

def generator(samples, batch_size=8):
	num_samples = len(samples)
	while 1:
		sklearn.utils.shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]
			augmented_images, augmented_measurements = [], []
			for batch_sample in batch_samples:
				steering_center = float(batch_sample[3]) #only steering

				# create adjusted steering measurements for the side camera images
				correction = 0.2 # this is a parameter to tune
				steering_left = steering_center + correction
				steering_right = steering_center - correction
				
				augmented_measurements.extend([steering_center, steering_left, steering_right])
				augmented_measurements.extend([steering_center*-1.0]) #flip steer
				
				
				#center camera
				source_path = batch_sample[0] 
				filename = source_path.split('/')[-1]
				current_path = '../data/IMG/' + filename
				image_center = cv2.imread(current_path)
				
				#left camera
				source_path = batch_sample[1] 
				filename = source_path.split('/')[-1]
				current_path = '../data/IMG/' + filename
				image_left = cv2.imread(current_path)
				
				#right camera
				source_path = batch_sample[2] 
				filename = source_path.split('/')[-1]
				current_path = '../data/IMG/' + filename
				image_right = cv2.imread(current_path)
				
				
				augmented_images.extend([image_center, image_left, image_right])
				augmented_images.extend([cv2.flip(image_center,1)]) #flip image

			# trim image to only see section with road
			X_train = np.array(augmented_images)
			y_train = np.array(augmented_measurements)
			yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=8)
validation_generator = generator(validation_samples, batch_size=8)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Convolution2D, MaxPooling2D, Cropping2D, Dropout

#LeNet
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
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
history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, \
									 nb_val_samples=len(validation_samples), nb_epoch=2)
model.save('lenet_model.h5')


print(history_object.history.keys())
print('Loss')
print(history_object.history['loss'])
print('Validation Loss')
print(history_object.history['val_loss'])


import matplotlib.pyplot as plt

plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig("lenet_loss.jpg")
'''
Epoch 1/2
6428/6428 [==============================] - 245s 38ms/step - loss: 0.0285 - val_loss: 0.0265
Epoch 2/2
6428/6428 [==============================] - 243s 38ms/step - loss: 0.0153 - val_loss: 0.0238
'''
