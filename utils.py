from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Activation, Flatten, Dense, Dropout, BatchNormalization, Concatenate
from tensorflow.keras.utils import plot_model
from collections import defaultdict
import visualkeras
import matplotlib
import plotly.graph_objects as graph
import imutils
from PIL import ImageFont
import os
from drig.config import logging as log
from drig.config import Error, ImageFontPath
import json
import glob
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.preprocessing import LabelEncoder
import plotly.figure_factory as ff


def display_image(
    image_path: str = None,
    image: np.ndarray = None,
    resize_ratio: int = None,
):
    try:

        if image_path:
            image = cv2.imread(image_path)
            if type(image) != np.ndarray:
                raise OSError(f"{Error.IMAGE_PATH_ERROR} : {image_path}")
        if resize_ratio:
            image = imutils.resize(
                image,
                width=image.shape[1] * resize_ratio,
            )
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    except Exception as e:
        raise e


def plot_training_metrics(
    model_training_history=None,
    epochs: int = None,
    json_path: str = None,
    save_path: str = None,
    use_matplotlib: bool = False,
    callback: bool = False,
    inline: bool = False,
):
    try:

        if json_path:
            json_file = open(json_path, mode="r")
            model_training_history = json.load(json_file)
        elif epochs is not None and model_training_history:
            pass
        else:
            raise Exception(Error.TRAINING_METRICS_PLOT_ERROR)

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


def matplotlib_plot(
    plot_values: tuple,
    callback: bool,
    inline: bool,
    save_path: str,
):
    try:
        x_axis, *loss_accuracy_value_list = plot_values
        y_training_loss, y_validation_loss, y_training_accuracy, y_validation_accuracy = loss_accuracy_value_list
        plt.style.use("ggplot")
        plt.figure(figsize=(8, 6), dpi=80)
        plt.plot(x_axis, y_training_loss, label="Training Loss")
        plt.plot(x_axis, y_validation_loss, label="Validation Loss")
        plt.plot(x_axis, y_training_accuracy, label="Training Accuracy")
        plt.plot(x_axis, y_validation_accuracy, label="Validation Accuracy")
        plt.title(f"Training/Validation Loss & Accuracy : EPOCH {len(x_axis)}")
        plt.xlabel("EPOCH #")
        plt.ylabel("LOSS/ACCURACY")
        plt.legend()
        if save_path:
            plt.savefig(save_path)
        if not callback and inline:
            plt.show()
    except Exception as e:
        raise e


def display_prediction(
    prediction: np.ndarray,
    class_labels: list,
    image: np.ndarray = None,
    image_path: str = None,
    text_origin: tuple = (10, 30),
    font_scale: float = 0.5,
    text_color: tuple = (0, 255, 0),
    text_thickness: int = 1,
    resize_ratio: int = None,
    text: str = None,
):
    try:
        if image_path:
            image = read_image(image_path)
        elif image.any():
            pass
        else:
            raise Exception(Error.NO_IMAGE_OR_PATH_ERROR)
        if resize_ratio:
            image = imutils.resize(
                image,
                width=image.shape[1] * resize_ratio,
            )

        return display_image(image=cv2.putText(
            image,
            f"Class: {class_labels[prediction[0]]}",
            text_origin,
            cv2.FONT_HERSHEY_COMPLEX,
            font_scale,
            text_color,
            text_thickness,
        ))
    except Exception as e:
        raise e


def grab_ranked_accuracies(
    predictions: np.ndarray,
    labels: list,
):
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


def visualize_network(
    network,
    scale_xy: int = 2,
    spacing: int = 10,
    scale_z=0.1,
    save_image_path: str = None,
):
    try:
        font = ImageFont.truetype(ImageFontPath.SKIA, 16)

        color_map = defaultdict(dict)
        color_map[Conv2D]['fill'] = '#CA6F1E'
        color_map[Activation]['fill'] = "#662851"
        color_map[Dropout]['fill'] = '#212F3D'
        color_map[MaxPooling2D]['fill'] = '#006fC1'
        color_map[Dense]['fill'] = '#145A32'
        color_map[Flatten]['fill'] = '#229954'
        color_map[BatchNormalization]['fill'] = '#BDC3C7'
        color_map[AveragePooling2D]['fill'] = '#4EACF2'
        color_map[Concatenate]['fill'] = "#4A235A"
        return visualkeras.layered_view(
            network,
            color_map=color_map,
            font=font,
            legend=True,
            scale_xy=scale_xy,
            spacing=spacing,
            to_file=save_image_path,
            scale_z=scale_z,
        )
    except Exception as e:
        raise e


def display_image_data(
    image_dataset_path: str,
    image_dim: tuple = None,
    tabs: int = 4,
):
    try:
        if not os.path.exists(image_dataset_path):
            raise OSError(f"{Error.DATASET_PATH_ERROR} : {image_dataset_path}")
        image_paths = list(imutils.paths.list_images(image_dataset_path))
        image_list = [
            Image.open(image_path).resize(
                (image_dim)) if image_dim else Image.open(image_path)
            for image_path in np.random.choice(image_paths, tabs)
        ]
        return visualkeras.utils.linear_layout(image_list)
    except Exception as e:
        raise e


def draw_faces(
    image: np.ndarray,
    faces: list,
):
    try:
        _ = [
            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                1,
            ) for x1, y1, x2, y2 in faces
        ]

        return image
    except Exception as e:
        raise e


def plot(
    plot_values: tuple,
    callback: bool,
    inline: bool,
    save_path: str,
):
    try:

        x_axis, *loss_accuracy_value_list = plot_values
        y_training_loss, y_validation_loss, y_training_accuracy, y_validation_accuracy = loss_accuracy_value_list
        fig = graph.Figure()
        fig.add_trace(
            graph.Scatter(
                x=x_axis,
                y=y_training_loss,
                mode='lines',
                name='Training Loss',
            ))
        fig.add_trace(
            graph.Scatter(
                x=x_axis,
                y=y_validation_loss,
                mode='lines',
                name='Validation Loss',
            ))
        fig.add_trace(
            graph.Scatter(
                x=x_axis,
                y=y_training_accuracy,
                mode='lines',
                name='Training Accuracy',
            ))
        fig.add_trace(
            graph.Scatter(
                x=x_axis,
                y=y_validation_accuracy,
                mode='lines',
                name='Validation Accuracy',
            ))

        fig.update_layout(
            autosize=False,
            plot_bgcolor="#d9d9d9",
            title=f"Training/Validation Loss & Accuracy : EPOCH {len(x_axis)}",
            width=1000,
            height=600,
            xaxis_title="EPOCH #",
            yaxis_title="LOSS/ACCURACY",
        )

        if save_path:
            fig.write_image(save_path)

        if not callback and inline:
            fig.show()

    except Exception as e:
        raise e


def plot_network(network):
    try:
        return plot_model(
            network,
            show_dtype=True,
            show_shapes=True,
        )
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


def grab_image_cast(image_path: str):
    try:
        image = cv2.imread(image_path)
        if type(image) != np.ndarray:
            raise OSError(f"{Error.IMAGE_PATH_ERROR} : {image_path}")
        return image.shape
    except Exception as e:
        raise e


def grab_image_class(
    image_path: str,
    class_index: int,
):
    try:
        class_name = image_path.replace(".", " ").replace(
            "/", " ").split(" ")[class_index]
        return class_name
    except Exception as e:
        raise e


def grab_confusion_mesh(
    tenets: np.ndarray,
    predictions: np.ndarray,
    encoded_classes: np.ndarray = None,
    class_name: str = None,
    class_index: int = None,
    classes: np.ndarray = None,
):
    try:
        confusion_mesh = multilabel_confusion_matrix(
            tenets,
            predictions,
            labels=encoded_classes,
        )

        if (class_index or class_name) and classes:
            if class_index:
                class_confusion_mesh = confusion_mesh[class_index]
                class_name = classes[class_index]

            elif class_name:
                class_index = list(classes).index(class_name)
                class_confusion_mesh = confusion_mesh[class_index]

            class_confusion_mesh[0, 0], class_confusion_mesh[
                1, 1] = class_confusion_mesh[1, 1], class_confusion_mesh[0, 0]
            return class_confusion_mesh, class_name, class_index
        return confusion_mesh
    except Exception as e:
        raise e


def plot_confusion_mesh(
    confusion_mesh: np.ndarray,
    class_name: str = None,
    save_path: str = None,
):
    try:

        hover_text = [["FALSE NEGATIVES", "TRUE NEGATIVES"],
                      ["TRUE POSITIVES", "FALSE POSITIVES"]]

        truth_values = ["TRUE", "FALSE"]

        scaled_values = np.sort(
            np.interp(
                confusion_mesh,
                (confusion_mesh.min(), confusion_mesh.max()),
                (0, +1),
            ).reshape(1, -1)).squeeze().tolist()
        colors = ["#822e5f", "#662851", "#660f53", "#47003c"]
        custom_color_scale = list(zip(scaled_values, colors))

        mesh_title = f"<i><b>CONFUSION MATRIX</b></i> : {class_name}"
        fig = ff.create_annotated_heatmap(
            confusion_mesh,
            x=truth_values,
            y=truth_values,
            colorscale=custom_color_scale,
            text=hover_text,
            hoverinfo='text',
        )
        fig.update_layout(
            title_text=mesh_title,
            xaxis=dict(title="TRUE VALUE"),
            yaxis=dict(
                title="PREDICTED VALUE",
                categoryorder='category ascending',
            ),
        )
        if save_path:
            fig.write_image(save_path)

        fig.show()

    except Exception as e:
        raise e


def grab_image_paths(dataset_path: str = None):
    try:
        all_image_paths = list(imutils.paths.list_images(dataset_path))
        if not os.path.exists(dataset_path):
            raise OSError(f"{Error.DATASET_PATH_ERROR} : {dataset_path}")
        if not all_image_paths:
            raise Exception(f"{Error.EMPTY_DATASET_ERROR} : {dataset_path}")
        return all_image_paths
    except Exception as e:
        raise e


def grab_random_image(
    dataset_path: str = None,
    image_paths: list = None,
    class_index: int = None,
    return_image_path: bool = False,
):
    try:
        if dataset_path:
            all_image_paths = grab_image_paths(dataset_path)
        elif image_paths:
            all_image_paths = image_paths
        random_image_path = np.random.choice(all_image_paths)
        image = read_image(random_image_path)
        upshots = (image, )

        if class_index:
            class_name = grab_image_class(
                random_image_path,
                class_index,
            )
            upshots = (
                *upshots,
                class_name,
            )

        if return_image_path:
            upshots = (
                *upshots,
                random_image_path,
            )

        return upshots
    except Exception as e:
        raise e


def read_image(image_path: str):
    try:
        return cv2.imread(image_path)
    except Exception as e:
        raise e


def preprocess_image(
    preprocessors: list,
    image: np.ndarray = None,
    image_path: str = None,
    for_prediction: bool = False,
):
    try:
        if image.any():
            pass
        elif image_path:
            image = read_image(image_path)
        else:
            raise Exception(Error.NO_IMAGE_OR_PATH_ERROR)
        for preprocessor in preprocessors:
            image = preprocessor.preprocess(image)
        if for_prediction:
            image = np.expand_dims(image, axis=0)
        return image
    except Exception as e:
        raise e


def grab_image_class_names(
    dataset_path: str = None,
    image_paths: list = None,
    class_index: int = -3,
    encode_classes: bool = False,
):
    try:
        if dataset_path:
            image_paths = grab_image_paths(dataset_path)
        elif image_paths:
            pass
        else:
            raise Exception(Error.NO_DATASET_OR_IMAGE_PATHS_ERROR)
        class_names = np.unique([
            grab_image_class(
                image_path,
                class_index,
            ) for image_path in image_paths
        ])

        if encode_classes:
            encoded_class_names = encode_class_names(class_names)
            return (
                class_names,
                encoded_class_names,
            )
        return class_names

    except Exception as e:
        raise e


def encode_class_names(class_names: np.ndarray):
    try:
        if class_names.size:
            encode_class_names = LabelEncoder().fit_transform(class_names)
            return encode_class_names
        raise Exception(Error.INVALID_PARAM_ERROR)
    except Exception as e:
        raise e
