from enum import Enum
import numpy as np
import json
import os
import progressbar
import cv2
from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from drig.feature.feature import FeatureCondenser
from drig.preprocessors.preprocessors import UniformAspectPreprocessor


class Hdf5DatumBuilder:
    def __init__(self, config: Enum):
        try:
            self.config = config
            self.image_paths = list(paths.list_images(
                config.IMAGES_PATH.value))
            self.labels = [
                config.LABEL_FROM_PATH(image_path)
                for image_path in self.image_paths
            ]
        except Exception as e:
            raise e

    def enconde_labels(self):
        try:
            label_encoder = LabelEncoder()
            return label_encoder.fit_transform(self.labels)
        except Exception as e:
            raise e

    def train_val_test_split(self):
        try:
            train_x_paths, test_x_paths, train_y_paths, test_y_paths = train_test_split(
                self.image_paths,
                self.labels,
                test_size=self.config.NUM_TESTING_IMAGES.value,
                random_state=42,
                stratify=self.labels)

            train_x_paths, val_x_paths, train_y_paths, val_y_paths = train_test_split(
                train_x_paths,
                train_y_paths,
                test_size=self.config.NUM_VALIDATION_IMAGES.value,
                random_state=42,
                stratify=self.labels)

            split_set = dict(
                train=(train_x_paths, train_y_paths,
                       self.config.TRAINING_DATUM_PATH.value),
                validation=(val_x_paths, val_y_paths,
                            self.config.VALIDATION_DATUM_PATH.value),
                test=(test_x_paths, test_y_paths,
                      self.config.TESTING_DATUM_PATH.value))
            return split_set
        except Exception as e:
            raise e
