import tensorflow as tf
import os
import logging

def get_data(validation_datasize):
    """ create a validation data set from the full training data

    Args:
        validation_datasize (int): validation data size
        
    Returns:
        (tuple): returns tuple containing training, validation and data size
    """
    logging.info("Preparing the data by segregating the training, validation and test data")
    
    mnist = tf.keras.datasets.mnist
    (X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()

    # create a validation data set from the full training data 
    # Scale the data between 0 to 1 by dividing it by 255. as its an unsigned data between 0-255 range
    X_valid, X_train = X_train_full[:validation_datasize] / 255., X_train_full[validation_datasize:] / 255.
    y_valid, y_train = y_train_full[:validation_datasize], y_train_full[validation_datasize:]

    # scale the test set as well
    X_test = X_test / 255.

    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)