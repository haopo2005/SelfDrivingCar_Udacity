import csv
import os

#use generator yield keyword, more memory efficient
lines = []
with open('../data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

with open('../selfdriving2/driving_log.csv') as csvfile:
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
				image_center = cv2.cvtColor(image_center, cv2.COLOR_BGR2RGB)
				
				#left camera
				source_path = batch_sample[1] 
				filename = source_path.split('/')[-1]
				current_path = '../data/IMG/' + filename
				image_left = cv2.imread(current_path)
				image_left = cv2.cvtColor(image_left, cv2.COLOR_BGR2RGB)
				
				#right camera
				source_path = batch_sample[2] 
				filename = source_path.split('/')[-1]
				current_path = '../data/IMG/' + filename
				image_right = cv2.imread(current_path)
				image_right = cv2.cvtColor(image_right, cv2.COLOR_BGR2RGB)
				
				augmented_images.extend([image_center, image_left, image_right])
				augmented_images.extend([cv2.flip(image_center,1)]) #flip image

			# trim image to only see section with road
			X_train = np.array(augmented_images)
			y_train = np.array(augmented_measurements)
			yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Convolution2D, MaxPooling2D, Cropping2D, Dropout

#LeNet
model = Sequential()
model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=(160,320,3)))
#((top_crop, bottom_crop), (left_crop, right_crop))
model.add(Cropping2D(cropping=((50,20),(0,0))))
model.add(Convolution2D(24,5,5, subsample = (2,2), activation = "relu"))
model.add(Dropout(.2))
model.add(Convolution2D(36,5,5, subsample = (2,2), activation = "relu"))
model.add(Dropout(.2))
model.add(Convolution2D(48,5,5, subsample = (2,2), activation = "relu"))
model.add(Dropout(.2))
model.add(Convolution2D(64,3,3, activation = "relu"))
model.add(Dropout(.2))
model.add(Convolution2D(64,3,3, activation = "relu"))
model.add(Dropout(.2))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(.2))#舍弃率设置为20％，这意味着从每个更新周期中随机排除10个输入中的2个。
model.add(Dense(50))
model.add(Dropout(.4))
model.add(Dense(10))
model.add(Dropout(.4))
model.add(Dense(1))
#it's regression problem, not classification 
#here we use mean square error loss function, no more softmax cross-entropy
model.compile(loss='mse', optimizer='adam')
model.summary()
history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples), 
									 validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=10)
model.save('nvidia.h5')

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
plt.savefig("nvidia_loss.jpg")