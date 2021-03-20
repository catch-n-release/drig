from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np


def display_image(image):
    try:
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    except Exception as e:
        raise e


def plot_training_metrics(epochs, model_training_history):
    try:
        x_axis = np.arange(0, epochs)
        plt.style.use("ggplot")
        plt.figure()
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
