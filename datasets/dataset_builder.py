from enum import Enum
import numpy as np
import json
import progressbar
import cv2
from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from drig.feature.condenser import FeatureCondenser
from drig.preprocessors.preprocessors import UniformAspectPreprocessor
from drig.config import log, DogsVsCatsConfig


class HDF5DatumBuilder:
    def __init__(self, config: Enum):
        try:
            log.info("-----INTIALIZING HDF5 DATASET BUILDER-----")
            self.config = config
            self.image_paths = list(paths.list_images(
                config.IMAGES_PATH.value))
            self.labels = [
                config.LABEL_FROM_PATH(image_path)
                for image_path in self.image_paths
            ]
            self.red, self.green, self.blue = list(), list(), list()
            self.preprocessing_height = 256
            self.preprocessing_width = 256
            self.preprocessing_depth = 3
            self.aspect_preprocessor = UniformAspectPreprocessor(
                self.preprocessing_height, self.preprocessing_width)
        except Exception as e:
            raise e

    def enconde_labels(self):
        try:
            log.info("-----ENCODING LABELS-----")
            label_encoder = LabelEncoder()
            self.labels = label_encoder.fit_transform(self.labels)
        except Exception as e:
            raise e

    def train_val_test_split(self):
        try:
            log.info(
                "-----SLITTING DATA INTO TRAINING VALIDATION & TESTING-----")
            train_x_paths, test_x_paths, train_y_lables, test_y_labels = train_test_split(
                self.image_paths,
                self.labels,
                test_size=self.config.NUM_TESTING_IMAGES.value,
                random_state=42,
                stratify=self.labels)

            train_x_paths, val_x_paths, train_y_lables, val_y_labels = train_test_split(
                train_x_paths,
                train_y_lables,
                test_size=self.config.NUM_VALIDATION_IMAGES.value,
                random_state=42,
                stratify=train_y_lables)

            split_set = dict(
                TRAINING=(train_x_paths, train_y_lables,
                          self.config.TRAINING_DATUM_PATH.value),
                VALIDATION=(val_x_paths, val_y_labels,
                            self.config.VALIDATION_DATUM_PATH.value),
                TESTING=(test_x_paths, test_y_labels,
                         self.config.TESTING_DATUM_PATH.value))
            return split_set
        except Exception as e:
            raise e

    def compose(self):
        try:

            log.info("-----BUILDING HDF5 DATASET-----")
            for set_type, (image_paths, labels,
                           save_dir) in self.train_val_test_split().items():
                condenser = FeatureCondenser(
                    (len(image_paths), self.preprocessing_height,
                     self.preprocessing_width, self.preprocessing_depth),
                    save_dir)
                widgets = [
                    f"BUILDING {set_type} DATUM :",
                    progressbar.Percentage(),
                    " ",
                    progressbar.Bar(marker="⎍"),
                    " ",
                    progressbar.SimpleProgress(),
                    " ",
                    progressbar.ETA(),
                ]
                prog_bar = progressbar.ProgressBar(maxval=len(image_paths),
                                                   widgets=widgets).start()

                for index, (image_path,
                            label) in enumerate(zip(image_paths, labels)):

                    image = cv2.imread(image_path)
                    preprocessed_image = self.aspect_preprocessor.preprocess(
                        image)
                    if set_type == "TRAINING":
                        b, g, r = cv2.mean(preprocessed_image)[:3]
                        self.red.append(r)
                        self.green.append(g)
                        self.blue.append(b)
                    condenser.commit([preprocessed_image], [label])
                    prog_bar.update(index)

            prog_bar.finish()
            condenser.lock()
            mean_rgb = dict(RED=np.mean(self.red),
                            GREEN=np.mean(self.green),
                            BLUE=np.mean(self.blue))
            log.info("-----WRITING MEAN RGB JSON FILE-----")
            json_file = open(self.config.MEAN_RGB.value, "w")
            json_file.write(json.dumps(mean_rgb))
            json_file.close()
        except TypeError:
            log.exception("---ERROR: LABELS NOT ENCODED----")
        except Exception as e:
            raise e
