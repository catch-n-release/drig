from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from collections import defaultdict
import visualkeras
import random
import matplotlib


def display_image(image):
    try:
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    except Exception as e:
        raise e


def plot_training_metrics(epochs, model_training_history, save_path=None):
    try:
        if save_path:
            matplotlib.use("Agg")
        x_axis = np.arange(0, epochs)
        plt.style.use("ggplot")
        plt.figure(figsize=(8, 6), dpi=80)
        plt.plot(x_axis,
                 model_training_history.history["loss"],
                 label="Training Loss")
        plt.plot(x_axis,
                 model_training_history.history["val_loss"],
                 label="Validation Loss")
        plt.plot(x_axis,
                 model_training_history.history["accuracy"],
                 label="Training Accuracy")
        plt.plot(x_axis,
                 model_training_history.history["val_accuracy"],
                 label="Validation Accuracy")
        plt.title("Training/Validation Loss & Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend()
        if save_path:
            plt.savefig(save_path)
        plt.show()
    except Exception as e:
        raise e


def display_prediction(image_path, prediction, class_labels):
    try:
        return display_image(
            cv2.putText(cv2.imread(image_path),
                        f"Label: {class_labels[prediction[0]]}", (10, 30),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2))
    except Exception as e:
        raise e


def get_ranked_accuracies(predictions, labels):
    try:
        rank_one_accuracy = 0
        rank_five_accuracy = 0

        for prediction, label in zip(predictions, labels):

            prediction = np.argsort(prediction)[::-1]

            if label in prediction[:5]:
                rank_five_accuracy += 1
            if label == prediction[0]:
                rank_one_accuracy += 1

        rank_one_accuracy = float(rank_one_accuracy / len(labels))
        rank_five_accuracy = float(rank_five_accuracy / len(labels))
        return rank_one_accuracy * 100, rank_five_accuracy * 100
    except Exception as e:
        raise e


def visualize_network(model, scale_xy=2):
    try:
        color_map = defaultdict(dict)
        color_map[Conv2D]['fill'] = '#CA6F1E'
        color_map[Activation]['fill'] = '#660000'
        color_map[Dropout]['fill'] = '#212F3D'
        color_map[MaxPooling2D]['fill'] = '#2471A3'
        color_map[Dense]['fill'] = '#145A32'
        color_map[Flatten]['fill'] = '#229954'
        color_map[BatchNormalization]['fill'] = '#BDC3C7'
        return visualkeras.layered_view(model,
                                        color_map=color_map,
                                        legend=True,
                                        scale_xy=scale_xy)
    except Exception as e:
        raise e


def display_data(image_paths):
    try:
        image_list = [
            Image.open(image_path)
            for image_path in random.sample(image_paths, 4)
        ]
        return visualkeras.utils.linear_layout(image_list)
    except Exception as e:
        raise e


def draw_faces(image, faces):
    try:
        _ = [
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
            for x1, y1, x2, y2 in faces
        ]

        return image
    except Exception as e:
        raise e
