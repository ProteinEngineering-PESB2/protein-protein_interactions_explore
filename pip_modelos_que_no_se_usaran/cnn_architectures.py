"""All CNN arquitectures"""
from math import ceil, sqrt
import tensorflow as tf
import numpy as np
from sklearn.metrics import (
    precision_score, accuracy_score, recall_score,
    f1_score, matthews_corrcoef, mean_squared_error,
    mean_absolute_error, r2_score, roc_auc_score, confusion_matrix
)
from scipy.stats import (kendalltau, pearsonr, spearmanr)
from keras.utils.layer_utils import count_params

class CnnA(tf.keras.models.Sequential):
    """
    mode = ("binary", "classification", "regression")

    Utiliza Convoluciones 1D, intercaladas con capas max pooling.

    Finaliza con flatten y dense.
    """
    def __init__(self, x_train, labels):
        super().__init__()
        
        self.add(tf.keras.layers.Conv1D(filters = 16, kernel_size = 2,
            activation="relu", input_shape=(x_train.shape[1], 1)))
        self.add(tf.keras.layers.MaxPool1D(pool_size = 2))

        self.add(tf.keras.layers.Conv1D(filters = 32, kernel_size = 3,
            activation="relu"))
        self.add(tf.keras.layers.AveragePooling1D(pool_size=4))

        self.add(tf.keras.layers.Conv1D(filters = 64, kernel_size = 2,
            activation="relu"))
        self.add(tf.keras.layers.MaxPool1D(pool_size=4))

        self.add(tf.keras.layers.Flatten())
        
        self.add(tf.keras.layers.Dense(units=64,
            activation="tanh"))
        
        self.add(tf.keras.layers.Dense(units=len(labels),
            activation="softmax"))
        self.compile(optimizer=tf.keras.optimizers.Adam(),
            loss = tf.keras.losses.SparseCategoricalCrossentropy())

class CnnB(tf.keras.models.Sequential):
    """
    mode = ("binary", "classification", "regression")

    Utiliza Convoluciones 1D, intercaladas con capas max pooling.

    Incorpora una capas Dropout con rate 0.25.

    Finaliza con flatten, y dense.
    """
    def __init__(self, x_train, labels):
        super().__init__()

        self.add(tf.keras.layers.Conv1D(filters = 16, kernel_size= 2, activation="relu",
            input_shape=(x_train.shape[1], 1)))
        self.add(tf.keras.layers.MaxPool1D(pool_size=2))
        self.add(tf.keras.layers.Dropout(0.25))

        self.add(tf.keras.layers.Conv1D(filters = 32, kernel_size = 3, activation="relu"))
        self.add(tf.keras.layers.AveragePooling1D(pool_size=4))
        self.add(tf.keras.layers.Dropout(0.25))
        
        self.add(tf.keras.layers.Conv1D(filters = 64, kernel_size = 2, activation="relu"))
        self.add(tf.keras.layers.MaxPool1D(pool_size=4))
        self.add(tf.keras.layers.Dropout(0.25))

        self.add(tf.keras.layers.Flatten())
        self.add(tf.keras.layers.Dense(units=64, activation="tanh"))

        self.add(tf.keras.layers.Dense(units=len(labels),
            activation="softmax"))
        self.compile(optimizer=tf.keras.optimizers.Adam(),
            loss = tf.keras.losses.SparseCategoricalCrossentropy())

class CnnC(tf.keras.models.Sequential):
    """
    mode = ("binary", "classification", "regression")

    Utiliza Dobles Convoluciones 1D, intercaladas con capas max pooling.

    Incorpora una capa Dropout con rate 0.25.

    Finaliza con flatten y dense.
    """
    def __init__(self, x_train, labels):
        super().__init__()

        self.add(tf.keras.layers.Conv1D(filters = 16, kernel_size= 2, activation="relu",
            input_shape=(x_train.shape[1], 1)))
        self.add(tf.keras.layers.Conv1D(filters = 16, kernel_size = 2, activation="relu"))
        self.add(tf.keras.layers.MaxPool1D(pool_size=2))
        self.add(tf.keras.layers.Dropout(0.25))

        self.add(tf.keras.layers.Conv1D(filters = 32, kernel_size = 3, activation="relu"))
        self.add(tf.keras.layers.Conv1D(filters = 32, kernel_size = 3, activation="relu"))
        self.add(tf.keras.layers.AveragePooling1D(pool_size=4))
        self.add(tf.keras.layers.Dropout(0.25))
        
        self.add(tf.keras.layers.Conv1D(filters = 64, kernel_size = 2, activation="relu"))
        self.add(tf.keras.layers.Conv1D(filters = 64, kernel_size = 2, activation="relu"))
        self.add(tf.keras.layers.MaxPool1D(pool_size=4))
        self.add(tf.keras.layers.Dropout(0.25))
    
        self.add(tf.keras.layers.Flatten())
        self.add(tf.keras.layers.Dense(units=64, activation="tanh"))
        self.add(tf.keras.layers.Dense(units=len(labels),
            activation="softmax"))
        self.compile(optimizer=tf.keras.optimizers.Adam(),
            loss = tf.keras.losses.SparseCategoricalCrossentropy())
        
class CnnD(tf.keras.models.Sequential):
    """
    mode = ("binary", "classification", "regression")

    Utiliza Dobles Convoluciones 1D, intercaladas con capas max pooling.

    Incorpora una capa Dropout con rate 0.25.

    Finaliza con flatten, y dense dividiendo las neuronas a la mitad por cada capa.
    """
    def __init__(self, x_train, labels, mode = "binary"):
        super().__init__()

        self.add(tf.keras.layers.Conv1D(filters = 16, kernel_size= 2, activation="relu",
            input_shape=(x_train.shape[1], 1)))
        self.add(tf.keras.layers.Conv1D(filters = 16, kernel_size = 2, activation="relu"))
        self.add(tf.keras.layers.MaxPool1D(pool_size=2))
        self.add(tf.keras.layers.Dropout(0.25))

        self.add(tf.keras.layers.Conv1D(filters = 32, kernel_size = 3, activation="relu"))
        self.add(tf.keras.layers.Conv1D(filters = 32, kernel_size = 3, activation="relu"))
        self.add(tf.keras.layers.AveragePooling1D(pool_size=4))
        self.add(tf.keras.layers.Dropout(0.25))
        
        self.add(tf.keras.layers.Conv1D(filters = 64, kernel_size = 2, activation="relu"))
        self.add(tf.keras.layers.Conv1D(filters = 64, kernel_size = 2, activation="relu"))
        self.add(tf.keras.layers.MaxPool1D(pool_size=4))
        self.add(tf.keras.layers.Dropout(0.25))

        self.add(tf.keras.layers.Flatten())

        unit = 64
        while unit > len(labels):
            self.add(tf.keras.layers.Dense(units=unit, activation="tanh"))
            unit = int(unit / 2)

        self.add(tf.keras.layers.Dense(units=len(labels),
            activation="softmax"))
        self.compile(optimizer=tf.keras.optimizers.Adam(),
            loss = tf.keras.losses.SparseCategoricalCrossentropy())

class Models:
    """Organize CNN objects, train and validation process"""
    def __init__(self, x_train, y_train, x_test, y_test, labels, arquitecture):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.labels = labels
        self.arquitecture = arquitecture


        if self.arquitecture == "A":
            self.cnn = CnnA(x_train=self.x_train, labels = self.labels)
        elif self.arquitecture == "B":
            self.cnn = CnnB(x_train=self.x_train, labels = self.labels)
        elif self.arquitecture == "C":
            self.cnn = CnnC(x_train=self.x_train, labels = self.labels)
        else:
            self.cnn = CnnD(x_train=self.x_train, labels = self.labels)
        
    def fit_models(self, epochs, verbose):
        """Fit model"""
        self.cnn.fit(self.x_train, self.y_train, epochs = epochs, verbose = verbose)

    def save_model(self, folder, prefix = ""):
        """
        Save model in .h5 format, in 'folder' location
        """
        self.cnn.save(f"{folder}/{prefix}-{self.arquitecture}-{self.mode}.h5")

    def get_metrics(self):
        """
        Returns classification performance metrics.

        Accuracy, recall, precision, f1_score, mcc.
        """
        trainable_count = count_params(self.cnn.trainable_weights)
        non_trainable_count = count_params(self.cnn.non_trainable_weights)
        result = {}
        result["arquitecture"] = self.arquitecture
        result["trainable_params"] = trainable_count
        result["non_trainable_params"] = non_trainable_count

        y_train_predicted = np.argmax(self.cnn.predict(self.x_train), axis = 1)
        y_test_score = self.cnn.predict(self.x_test)
        y_test_predicted = np.argmax(y_test_score, axis = 1)

        result["labels"] = self.labels
        train_metrics = {
            "accuracy": accuracy_score(y_true = self.y_train, y_pred = y_train_predicted),
            "recall": recall_score(
                y_true = self.y_train, y_pred = y_train_predicted, average="micro"),
            "precision": precision_score(
                y_true = self.y_train, y_pred = y_train_predicted, average="micro"),
            "f1_score": f1_score(
                y_true = self.y_train, y_pred = y_train_predicted, average="micro"),
            "mcc": matthews_corrcoef(y_true = self.y_train, y_pred = y_train_predicted),
            "confusion_matrix": confusion_matrix(
                y_true = self.y_train, y_pred = y_train_predicted).tolist()
        }
        test_metrics = {
            "accuracy": accuracy_score(y_true = self.y_test, y_pred = y_test_predicted),
            "recall": recall_score(
                y_true = self.y_test, y_pred = y_test_predicted, average="micro"),
            "precision": precision_score(
                y_true = self.y_test, y_pred = y_test_predicted, average="micro"),
            "f1_score": f1_score(
                y_true = self.y_test, y_pred = y_test_predicted, average="micro"),
            "mcc": matthews_corrcoef(y_true = self.y_test, y_pred = y_test_predicted),
            "confusion_matrix": confusion_matrix(
                y_true = self.y_test, y_pred = y_test_predicted).tolist()
        }

        result["train_metrics"] = train_metrics
        result["test_metrics"] = test_metrics
        return result