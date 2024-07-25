import datetime
import itertools
import os
import random
import zipfile
import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score


def view_random_image(target_dir, target_class):
    target_folder = target_dir + "/" + target_class

    random_image = random.sample(os.listdir(target_folder), 1)

    img = mpimg.imread(target_folder + "/" + random_image[0])

    plt.imshow(img, cmap='viridis', vmin=0, vmax=255)
    plt.title(target_class)
    plt.axis('off')
    print(f"Image shape is : {img.shape}")
    return img


def plot_loss_curves(history):
    """
    Returns separate loss curves for training and validation metrics from the history variable of ur model.
    Args:
        history: Tensorflow History Object
    Returns:
        Plots of validation and training loss and accuracy metrics.
    """

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    epochs = range(len(history.history["loss"]))

    plt.plot(epochs, loss, label="training_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    plt.figure()
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, val_accuracy, label="val_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()


def load_prepare_image_in_tf(filepath, img_shape=224, scale=True):
    """
    Reads in an image from filename, turns it into a tensor and reshapes it to
    (224, 224, 3).

    Parameters:
        filepath (str): String filepath of the target image
        img_shape (int): Size to resize the target image, default is 224*224
        scale (bool): Whether to scale the pixel values to range(0, 1), default is true.
    Returns:
        Tensor Image.
    """
    img = tf.io.read_file(filepath)
    img = tf.image.decode_image(img)
    img = tf.image.resize(img, [img_shape, img_shape])

    if scale:
        return img / 255.
    else:
        return img


def create_tensorboard_callback(dir_name, experiment_name):
    """
    Creates a tensorboard callback and saves it to the given directory with the given experiment name
    Args:
        dir_name (str): Name of the directory where u want to store the callback
        experiment_name (str): Name for the experiment you want to save it as
    Returns:
        Returns a Tensorboard Callback Object
    """
    log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    print(f"Saving tensorboard callbacks at :{log_dir}")
    return tensorboard_callback


def create_model_from_url(url, IMAGE_SHAPE=(224, 224), num_classes=10):
    """
    Takes a TF Hub URL and creates a keras sequential model with it
    Args:
        url(str) = A TF Hub Model URL
        num_classes (int) = Number of output neurons in the output layer, should be equal to number of target classes.
        IMAGE_SHAPE (int, int): The shape of the input for the model, default value is (224,224)
    Returns:
        An un-compiled keras sequential model from the url as feature extractor layer and dense output layer with num_Classes as output neurons.
    """
    feature_extractor_layer = tf_hub.KerasLayer(url,
                                                trainable=False,
                                                name="feature_extraction_layer",
                                                input_shape=IMAGE_SHAPE + (3,))

    model = Sequential([
        feature_extractor_layer,
        Dense(num_classes, activation="softmax", name="output_layer")
    ])

    return model


def unzip_data(file_path, save_dir="Datasets/"):
    """
    Unzips the zip file and saves it into the directory
    :param file_path: Path of the zip file to be extracted.
    :param save_dir: Path of the directory where the contents of the zip files are to be saved.
    :return: None
    """
    os.makedirs(save_dir, exist_ok=True)
    zip_ref = zipfile.ZipFile(file_path, "r")
    zip_ref.extractall(save_dir)
    return zip_ref


def plot_loss_curves_two(history1, history2, initial_epochs=5):
    """
    :param history1: History object of the first model.
    :param history2: History object of the second model.
    :param initial_epochs: initial epochs of the first model
    :return: Plots comparison graphs
    """

    acc = history1.history["accuracy"]
    loss = history1.history["loss"]

    val_acc = history2.history['accuracy']
    val_loss = history2.history['loss']

    total_acc = acc + history2.history['accuracy']
    total_loss = loss + history2.history['loss']

    total_val_acc = val_acc + history2.history['accuracy']
    total_val_loss = val_loss + history2.history['loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(total_acc, label="Training Accuracy")
    plt.plot(total_val_acc, label="Validation Accuracy")
    plt.plot([initial_epochs - 1, initial_epochs - 1], plt.ylim(), label="Start Fine Tuning")
    plt.legend(loc="lower right")
    plt.title("Training and Validation Accuracy")

    plt.subplot(2, 1, 2)
    plt.plot(total_loss, label="Training Loss")
    plt.plot(total_val_loss, label="Validation Loss")
    plt.plot([initial_epochs - 1, initial_epochs - 1], plt.ylim(), label="Start Fine Tuning")
    plt.legend(loc="upper right")
    plt.title("Training and Validation Loss")


def return_data_aug_layer_for_eff_net(random_flip="horizontal",
                                      random_rotation=0.2,
                                      random_zoom=0.2,
                                      random_width=0.2,
                                      random_height=0.2):
    """

    :param random_flip: Flip value, default = "horizontal.
    :param random_rotation: Rotation value, default = 0.2.
    :param random_zoom: Zoom value, default = 0.2.
    :param random_width: Width value, default = 0.2.
    :param random_height: Height value, default = 0.2.
    :return: Returns a data augmentation layer from keras.Sequential
    """

    data_augmentation = Sequential([
        preprocessing.RandomFlip(random_flip),
        preprocessing.RandomRotation(random_rotation),
        preprocessing.RandomWidth(random_width),
        preprocessing.RandomZoom(random_zoom),
        preprocessing.RandomHeight(random_height)
    ], name="Data_Augmentation")

    return data_augmentation


def make_confusion_matrix(y_true, y_pred, fig_size=(10, 10), text_size=15, classes=None, norm=False, save_fig=False):
    """
    Takes the true and predicted labels and prints the confusion matrix for all the classes in a beautiful way.
    :param text_size: Size of the text on the plot.
    :param fig_size: Size of the plot.
    :param save_fig: Saves the confusion matrix in image
    :param norm: Normalize the values or not
    :param classes: The classes in the labels
    :param y_true: True labels.
    :param y_pred: Predicted labels by your model
    :return: Prints a beautiful confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    n_classes = cm.shape[0]

    # Plot the figure and make it pretty
    fig, ax = plt.subplots(figsize=fig_size)
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    # Are there a list of classes?
    if classes:
        labels = classes
    else:
        labels = np.arange(cm.shape[0])

    # Label the axes
    ax.set(title="Confusion Matrix",
           xlabel="Predicted label",
           ylabel="True label",
           xticks=np.arange(n_classes),
           yticks=np.arange(n_classes),
           xticklabels=labels,
           yticklabels=labels)

    # Make x-axis labels appear on bottom
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    plt.xticks(rotation=70, fontsize=text_size)
    plt.yticks(fontsize=text_size)
    # Set the threshold for different colors
    threshold = (cm.max() + cm.min()) / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if norm:
            plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j] * 100:.1f}%)",
                     horizontalalignment="center",
                     color="white" if cm[i, j] > threshold else "black",
                     size=15)
        else:
            plt.text(j, i, f"{cm[i, j]}",
                     horizontalalignment="center",
                     color="white" if cm[i, j] > threshold else "black",
                     size=15)

    if save_fig:
        fig.savefig("ConfusionMatrix/confusion_matrix.png")


def calc_evaluation_metrics(y_true, y_pred):
    """

    :param y_true: True labels of the test dataset you have.
    :param y_pred: Predicted labels of the model you have built.
    :return: A dictionary of accuracy, recall, precision, f1-score of your model
    """

    results = dict()
    results["Accuracy"] = accuracy_score(y_true, y_pred)*100
    results["Precision"] = precision_score(y_true, y_pred, average="weighted")
    results["Recall"] = precision_score(y_true, y_pred, average="weighted")
    results["F1-Score"] = f1_score(y_true, y_pred, average="weighted")

    return results


def load_prep_image(image_name_path, img_size=None):
    """

    :param img_size: Size the image needs to be converted to, default is (224, 224)
    :param image_name_path: Path of the image to be processed
    :return: Returns the processed image

    """

    if img_size is None:
        img_size = [224, 224]

    img = tf.io.read_file(image_name_path)
    img = tf.image.decode_image(img)
    img = tf.expand_dims(img, axis=0)
    img = tf.image.resize(img, size=img_size)
    img = img / 255.

    return img


def add_top_three_row_values(dataframe: pd.DataFrame, y_labels: list):
    """

    :param dataframe: The dataframe of predicition probabilities
    :param y_labels: The true labels of the dataset
    :return: Returns a dataframe of two columns, one with a list of top 3 predictions and other with the true label
    """
    top_3_preds = pd.DataFrame()
    top3_lists = []
    for index, row in dataframe.iterrows():
        row_values = row.values
        top_indices = np.argsort(row_values)[-3:][::-1]
        top_columns = dataframe.columns[top_indices]
        top3_lists.append(top_columns.tolist())
    top_3_preds['top 3 cols'] = top3_lists
    top_3_preds['True Label'] = y_labels
    return top_3_preds