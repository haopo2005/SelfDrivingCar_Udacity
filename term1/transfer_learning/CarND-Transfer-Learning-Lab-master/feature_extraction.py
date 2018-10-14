import pickle
import tensorflow as tf
# TODO: import Keras layers you need here
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from sklearn.preprocessing import LabelBinarizer
import numpy as np
flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('training_file', '', "Bottleneck features training file (.p)")
flags.DEFINE_string('validation_file', '', "Bottleneck features validation file (.p)")


def load_bottleneck_data(training_file, validation_file):
    """
    Utility function to load bottleneck features.

    Arguments:
        training_file - String
        validation_file - String
    """
    print("Training file", training_file)
    print("Validation file", validation_file)

    with open(training_file, 'rb') as f:
        train_data = pickle.load(f)
    with open(validation_file, 'rb') as f:
        validation_data = pickle.load(f)

    X_train = train_data['features']
    y_train = train_data['labels']
    X_val = validation_data['features']
    y_val = validation_data['labels']

    return X_train, y_train, X_val, y_val


def main(_):
    # load bottleneck data
    X_train, y_train, X_val, y_val = load_bottleneck_data(FLAGS.training_file, FLAGS.validation_file)

    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)

    # TODO: define your model and hyperparams here
    # make sure to adjust the number of classes based on
    # the dataset
    # 10 for cifar10
    # 43 for traffic  
    label_binarizer = LabelBinarizer()
    y_train_hot = label_binarizer.fit_transform(y_train)
    y_val_hot = label_binarizer.fit_transform(y_val)
    
    
    model = Sequential()
    model.add(Flatten(input_shape=(1, 1, 512)))
    model.add(Dense(len(np.unique(y_train))))
    model.add(Activation('softmax'))
    # TODO: train your model here
    model.summary()
    #loss='sparse_categorical_crossentropy',可以不用手动求one hot
    model.compile('adam', 'categorical_crossentropy', ['accuracy'])
    history = model.fit(X_train, y_train_hot,batch_size=128, epochs=20, validation_data=(X_val,y_val_hot))

# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
