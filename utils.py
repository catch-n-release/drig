from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.core import Activation, Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers import Concatenate
from keras.utils import plot_model
from collections import defaultdict
import visualkeras
import matplotlib
import plotly.graph_objects as graph
from imutils import paths
from PIL import ImageFont
import os
from drig.config import logging as log
import json
import glob


def display_image(image_path: str = None, image: np.ndarray = None):
    try:

        if image_path:
            image = cv2.imread(image_path)
            assert type(image) == np.ndarray, "INVALID IMAGE PATH"
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    except Exception as e:
        raise e


def plot_training_metrics(model_training_history=None,
                          epochs: int = None,
                          json_path: str = None,
                          save_path: str = None,
                          use_matplotlib: bool = False,
                          callback: bool = False,
                          inline: bool = False):
    try:

        if json_path:
            json_file = open(json_path, mode="r")
            model_training_history = json.load(json_file)
        elif epochs and model_training_history:
            pass
        else:
            raise Exception(
                "EITHER JSON PATH OR EPOCHS & MODEL TRAINING HISTORY SHOULD BE SUPPLIED"
            )

        if callback or json_path:
            y_training_loss, y_validation_loss, y_training_accuracy, y_validation_accuracy = model_training_history[
                "loss"], model_training_history[
                    "val_loss"], model_training_history[
                        "accuracy"], model_training_history["val_accuracy"]

        else:
            y_training_loss, y_validation_loss, y_training_accuracy, y_validation_accuracy = model_training_history.history[
                "loss"], model_training_history.history[
                    "val_loss"], model_training_history.history[
                        "accuracy"], model_training_history.history[
                            "val_accuracy"]
        x_axis = np.arange(0, len(y_training_loss))
        plot_values = x_axis, y_training_loss, y_validation_loss, y_training_accuracy, y_validation_accuracy
        if use_matplotlib:
            if callback:
                matplotlib.use("Agg")
            # else:

            matplotlib_plot(plot_values, callback, inline, save_path)
        else:
            plot(plot_values, callback, inline, save_path=save_path)
    except Exception as e:
        raise e


def matplotlib_plot(plot_values, callback, inline, save_path):
    try:
        x_axis, *loss_accuracy_value_list = plot_values
        y_training_loss, y_validation_loss, y_training_accuracy, y_validation_accuracy = loss_accuracy_value_list
        plt.style.use("ggplot")
        plt.figure(figsize=(8, 6), dpi=80)
        plt.plot(x_axis, y_training_loss, label="Training Loss")
        plt.plot(x_axis, y_validation_loss, label="Validation Loss")
        plt.plot(x_axis, y_training_accuracy, label="Training Accuracy")
        plt.plot(x_axis, y_validation_accuracy, label="Validation Accuracy")
        plt.title("Training/Validation Loss & Accuracy")
        plt.xlabel("EPOCH #")
        plt.ylabel("LOSS/ACCURACY")
        plt.legend()
        if save_path:
            plt.savefig(save_path)
        if not callback and inline:
            plt.show()
    except Exception as e:
        raise e


def display_prediction(image_path, prediction, class_labels):
    try:
        return display_image(image=cv2.putText(
            cv2.imread(image_path), f"Label: {class_labels[prediction[0]]}",
            (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2), )
    except Exception as e:
        raise e


def ranked_accuracies(predictions, labels):
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


def visualize_network(model,
                      scale_xy: int = 2,
                      spacing: int = 10,
                      scale_z=0.1,
                      save_image_path: str = None):
    try:
        font = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Skia.ttf", 16)

        color_map = defaultdict(dict)
        color_map[Conv2D]['fill'] = '#CA6F1E'
        color_map[Activation]['fill'] = '#660000'
        color_map[Dropout]['fill'] = '#212F3D'
        color_map[MaxPooling2D]['fill'] = '#006fC1'
        color_map[Dense]['fill'] = '#145A32'
        color_map[Flatten]['fill'] = '#229954'
        color_map[BatchNormalization]['fill'] = '#BDC3C7'
        color_map[AveragePooling2D]['fill'] = '#4EACF2'
        color_map[Concatenate]['fill'] = "#4A235A"
        return visualkeras.layered_view(model,
                                        color_map=color_map,
                                        font=font,
                                        legend=True,
                                        scale_xy=scale_xy,
                                        spacing=spacing,
                                        to_file=save_image_path,
                                        scale_z=scale_z)
    except Exception as e:
        raise e


def display_image_data(image_datum_path: str, image_dim: tuple = None):
    try:
        if not os.path.exists(image_datum_path):
            raise Exception(f"Invalid Path : {image_datum_path}")
        image_paths = list(paths.list_images(image_datum_path))
        image_list = [
            Image.open(image_path).resize(
                (image_dim)) if image_dim else Image.open(image_path)
            for image_path in np.random.choice(image_paths, 4)
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


def plot(
    plot_values,
    callback,
    inline,
    save_path,
):
    try:

        x_axis, *loss_accuracy_value_list = plot_values
        y_training_loss, y_validation_loss, y_training_accuracy, y_validation_accuracy = loss_accuracy_value_list
        fig = graph.Figure()
        fig.add_trace(
            graph.Scatter(x=x_axis,
                          y=y_training_loss,
                          mode='lines',
                          name='Training Loss'))
        fig.add_trace(
            graph.Scatter(x=x_axis,
                          y=y_validation_loss,
                          mode='lines',
                          name='Validation Loss'))
        fig.add_trace(
            graph.Scatter(x=x_axis,
                          y=y_training_accuracy,
                          mode='lines',
                          name='Training Accuracy'))
        fig.add_trace(
            graph.Scatter(x=x_axis,
                          y=y_validation_accuracy,
                          mode='lines',
                          name='Validation Accuracy'))

        fig.update_layout(autosize=False,
                          plot_bgcolor="#d9d9d9",
                          title="Training/Validation Loss & Accuracy",
                          width=1000,
                          height=600,
                          xaxis_title="EPOCH #",
                          yaxis_title="LOSS/ACCURACY")

        if save_path:
            fig.write_image(save_path)

        if not callback and inline:
            fig.show()

    except Exception as e:
        raise e


def plot_network(net):
    try:
        return plot_model(net, show_dtype=True, show_shapes=True)
    except Exception as e:
        raise e


def compose_image_collages(
    images_path: str,
    collage_dim: tuple,
    indices: list,
    output_path: str = None,
    normalize_data=True,
):
    try:
        collage_images = list()
        for index in indices:
            class_index = index + 1
            class_root_path = os.path.join(images_path, f"{class_index}_*.jpg")
            class_image_paths = sorted(list(glob.glob(class_root_path)))
            collage_canvas = np.zeros(collage_dim, dtype="uint8")
            images_per_dim = (len(class_image_paths) // 2)
            resize_image_height = collage_dim[0] // images_per_dim
            resize_image_width = collage_dim[1] // images_per_dim
            images_per_class = list()
            for image_path in class_image_paths:
                image = cv2.imread(image_path)
                image = cv2.resize(image,
                                   (resize_image_height, resize_image_width))
                images_per_class.append(image)
            collage_canvas[0:resize_image_height,
                           0:resize_image_width] = images_per_class[2]
            collage_canvas[
                0:resize_image_height,
                resize_image_width:collage_dim[1]] = images_per_class[3]
            collage_canvas[resize_image_height:collage_dim[0],
                           0:resize_image_width] = images_per_class[1]
            collage_canvas[
                resize_image_height:collage_dim[0],
                resize_image_width:collage_dim[1]] = images_per_class[0]
            if output_path:
                os.makedirs(output_path, exist_ok=True)
                cv2.imwrite(os.path.join(output_path, f"{class_index}.png"),
                            collage_canvas)
            collage_images.append(collage_canvas)
        collage_images = np.array(collage_images)

        if normalize_data:
            collage_images = collage_images / 255.0

        return collage_images
    except Exception as e:
        raise e


def image_dim(image_path):
    try:
        image = cv2.imread(image_path)
        assert type(image) == np.ndarray, "INVALID IMAGE PATH"
        return image.shape
    except Exception as e:
        raise e
