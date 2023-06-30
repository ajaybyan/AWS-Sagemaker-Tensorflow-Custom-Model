import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Reshape
import argparse
import os
import numpy as np
import json
from s3fs.core import S3FileSystem


def model(x_train, y_train, epochs=1):
    
    model = Sequential()
    
    model.add(Reshape((28,28,1), input_shape = (28,28)))
    
    model.add(Conv2D(64, (5, 5),
                      padding="same",
                      activation="relu",
                      input_shape=(28, 28, 1)))
     
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (5, 5), padding="same",
                      activation="relu"))
     
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, (5, 5), padding="same",
                      activation="relu"))
     
    model.add(MaxPooling2D(pool_size=(2, 2)))
     
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.4))
     
    model.add(Dense(10, activation="softmax"))
    

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    
    model.fit(x_train, y_train, 
              batch_size = 64,
              epochs=epochs,
              validation_split=0.33)

    return model


def _load_training_data(base_dir):
    x_train = np.load(os.path.join(base_dir, "x_train.npy"))
    y_train = np.load(os.path.join(base_dir, "y_train.npy"))
    #s3 = S3FileSystem()
    
    #x_train = np.load(s3.open(os.path.join(base_dir, "x_train.npy")))
    #y_train = np.load(s3.open(os.path.join(base_dir, "y_train.npy")))
    
    return x_train, y_train


def _load_testing_data(base_dir):
    x_test = np.load(os.path.join(base_dir, "x_test.npy"))
    y_test = np.load(os.path.join(base_dir, "y_test.npy"))
                      
    #s3 = S3FileSystem()
    
    #x_test = np.load(s3.open(os.path.join(base_dir, "x_test.npy")))
    #y_test = np.load(s3.open(os.path.join(base_dir, "y_test.npy")))
                     
    return x_test, y_test



def _parse_args():
    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ.get('SM_HOSTS')))
    parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST'))
    parser.add_argument('--epochs', type=int, default=1)

    return parser.parse_known_args()


# List devices available to TensorFlow
from tensorflow.python.client import device_lib

if __name__ == "__main__":
    args, unknown = _parse_args()

    print(args)
    #print('SN_MODEL_DIR: {}\n\n'.format(args.SM_MODEL_DIR))
    
    print('\n\nDEVICES\n\n')    
    print(device_lib.list_local_devices())
    print('\n\n')
    
    print('Loading Fashion MNIST data..\n')
    train_data, train_labels = _load_training_data(args.train)
    eval_data, eval_labels = _load_testing_data(args.test)

    print('Training model for {} epochs..\n\n'.format(args.epochs))
    mnist_classifier = model(train_data, train_labels, epochs=args.epochs)
    
    # perform evaluation on test data
    test_eval = mnist_classifier.evaluate(eval_data, eval_labels)
    print(test_eval)

    if args.current_host == args.hosts[0]:
        # save model to an S3 directory with keras format
        mnist_classifier.save(os.path.join(args.sm_model_dir, 'my_model.h5'))
        
        # save model to an S3 directory with SaveModel format
        mnist_classifier.save(os.path.join(args.sm_model_dir, 'mnist_fashion_classifier'))
                              
