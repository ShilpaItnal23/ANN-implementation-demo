import tensorflow as tf
import time
import os
import matplotlib.pyplot as plt
import pandas as pd
import logging

def create_model(LOSS_FUNCTION, OPTIMIZER, METRICS, NUM_CLASSES):
    """[summary]

    Args:
        LOSS_FUNCTION (str): sparse_categorical_crossentropy
        OPTIMIZER (str): SGD
        METRICS (float): accuracy
        NUM_CLASSES (int): 10

    Returns:
        [python object]: untrained model
    """
    
    LAYERS = [tf.keras.layers.Flatten(input_shape=[28, 28], name="inputLayer"),
          tf.keras.layers.Dense(300, activation="relu", name="hiddenLayer1"),
          tf.keras.layers.Dense(100, activation="relu", name="hiddenLayer2"),
          tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="outputLayer")]

    model_clf = tf.keras.models.Sequential(LAYERS)

    model_clf.summary()

    model_clf.compile(loss=LOSS_FUNCTION,
                optimizer=OPTIMIZER,
                metrics=METRICS)

    return model_clf ## <<< untrained model
    logging.info(f"untrained model {model_clf}")
    
def get_unique_filename(filename):
    """path to save the model

    Args:
        filename (str): generates unique file name every time when it runs

    Returns:
        [str]: returns file name
    """
    unique_filename = time.strftime(f"%Y%m%d_%H%M%S_{filename}")
    return unique_filename
    logging.info(f"returns unique filename {unique_filename}")
    
def save_model(model, model_name, model_dir):
    """saving of the model with unique name

    Args:
        model (str): file name
        model_name (object): model.h5
        model_dir (objec): model
    """
        
    unique_filename = get_unique_filename(model_name)
    path_to_model = os.path.join(model_dir, unique_filename)
    model.save(path_to_model)

def save_plot(loss_acc,plot_name,plot_dir):
    """
    : param loss_acc: its a loss accuracy
    : param plot_name: it is name of the plot
    : param plot_dir: its directory to save the plot
    
    """
    logging.info("creating the  plot")

    unique_filename= get_unique_filename(plot_name)
    plotPath = os.path.join(plot_dir,unique_filename)
    pd.DataFrame(loss_acc).plot(figsize=(8,5))
    plt.grid(True)
    plt.gca().set_ylim(0,1)
    plt.savefig(plotPath)
    plt.show()

     