import numpy as np
import json
import progressbar
import cv2
from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from drig.feature import FeatureCondenser
from drig.preprocessors import UniformAspectPreprocessor
from drig.utils import log
import os
from dataclasses import dataclass


class HDF5DatumComposer:
    def __init__(self, config: dataclass):
        try:
            log.info("-----INTIALIZING HDF5 DATASET BUILDER-----")
            self.config = config
            self.hdf5_datum_path = self.config.HDF5_DATUM_PATH
            os.makedirs(self.hdf5_datum_path, exist_ok=True)
            self.image_paths = list(
                paths.list_images(config.TRAINING_IMAGES_PATH))
            self.labels = [
                image_path.replace(".", " ").replace(
                    "/", " ").split(" ")[config.TRAINING_LABEL_INDEX]
                for image_path in self.image_paths
            ]
            self.red, self.green, self.blue = list(), list(), list()
            self.preprocessing_height = self.config.IMAGE_PREPROCESSING_HEIGHT
            self.preprocessing_width = self.config.IMAGE_PREPROCESSING_WIDTH
            self.preprocessing_depth = self.config.IMAGE_PREPROCESSING_DEPTH
            self.aspect_preprocessor = UniformAspectPreprocessor(
                self.preprocessing_height, self.preprocessing_width)
        except Exception as e:
            raise e

    def enconde_labels(self):
        try:
            log.info("-----ENCODING LABELS-----")
            label_encoder = LabelEncoder()
            self.encoded_labels = label_encoder.fit_transform(self.labels)
        except Exception as e:
            raise e

    def train_val_test_split(self, train_test_only: bool = False):
        try:

            self.enconde_labels()
            train_x_paths, test_x_paths, train_y_lables, test_y_labels = train_test_split(
                self.image_paths,
                self.encoded_labels,
                test_size=self.config.NUM_TESTING_IMAGES,
                random_state=42,
                stratify=self.encoded_labels)
            log.info("-----SLITTING DATA INTO TRAINING & TESTING SET-----")
            if train_test_only:
                split_set = dict(TRAINING=(train_x_paths, train_y_lables,
                                           self.config.TRAINING_DATUM_PATH),
                                 TESTING=(test_x_paths, test_y_labels,
                                          self.config.TESTING_DATUM_PATH))
                return split_set

            train_x_paths, val_x_paths, train_y_lables, val_y_labels = train_test_split(
                train_x_paths,
                train_y_lables,
                test_size=self.config.NUM_VALIDATION_IMAGES,
                random_state=42,
                stratify=train_y_lables)
            log.info(
                "-----SLITTING DATA INTO TRAINING VALIDATION & TESTING SET-----"
            )
            split_set = dict(TRAINING=(train_x_paths, train_y_lables,
                                       self.config.TRAINING_DATUM_PATH),
                             VALIDATION=(val_x_paths, val_y_labels,
                                         self.config.VALIDATION_DATUM_PATH),
                             TESTING=(test_x_paths, test_y_labels,
                                      self.config.TESTING_DATUM_PATH))
            return split_set
        except Exception as e:
            raise e

    def compose(self, custom_split: dict = None):
        try:

            log.info("-----BUILDING HDF5 DATASET-----")
            log.info("----USING CUSTOM SPLIT---")
            split = custom_split
            if not custom_split:
                log.info("----CUSTOM SPLIT NOT PROVIDED---")
                log.info("----USING CONFIG SPLIT---")
                split = self.train_val_test_split()
            for set_type, (image_paths, labels, save_dir) in split.items():
                condenser = FeatureCondenser(
                    (len(image_paths), self.preprocessing_height,
                     self.preprocessing_width, self.preprocessing_depth),
                    save_dir)
                widgets = [
                    f"BUILDING {set_type} DATUM :",
                    progressbar.Percentage(),
                    " ",
                    progressbar.Bar(marker="❆"),
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
                    image = self.aspect_preprocessor.preprocess(image)
                    if set_type == "TRAINING":
                        b, g, r = cv2.mean(image)[:3]
                        self.red.append(r)
                        self.green.append(g)
                        self.blue.append(b)
                    condenser.commit([image], [label])
                    prog_bar.update(index)

            prog_bar.finish()
            condenser.latch()
            mean_rgb = dict(RED=np.mean(self.red),
                            GREEN=np.mean(self.green),
                            BLUE=np.mean(self.blue))
            log.info("-----WRITING MEAN RGB JSON FILE-----")
            with open(self.config.MEAN_RGB_PATH, "w") as json_file:
                json_file.write(json.dumps(mean_rgb))
        except TypeError:
            log.exception("---ERROR: LABELS NOT ENCODED----")
        except Exception as e:
            raise e
