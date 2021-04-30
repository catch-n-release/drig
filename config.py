from dataclasses import dataclass
import os
import logging

logging.basicConfig(level=logging.INFO)


@dataclass(frozen=True)
class DogsVsCatsConfig:
    DATASET_PATH: str = os.path.abspath(
        os.path.join(os.path.pardir, "datasets/kag_dogs_vs_cats"))
    IMAGES_PATH: str = os.path.join(DATASET_PATH, "train")
    HDF5_DATUM_PATH: str = os.path.join(DATASET_PATH, "hdf5_datum")
    TRAINING_DATUM_PATH: str = os.path.join(HDF5_DATUM_PATH,
                                            "training_datum.hdf5")
    VALIDATION_DATUM_PATH: str = os.path.join(HDF5_DATUM_PATH,
                                              "validation_datum.hdf5")
    TESTING_DATUM_PATH: str = os.path.join(HDF5_DATUM_PATH,
                                           "testing_datum.hdf5")

    NUM_CLASSES: int = 2
    NUM_VALIDATION_IMAGES: int = 1250 * NUM_CLASSES
    NUM_TESTING_IMAGES: int = 1250 * NUM_CLASSES

    IMAGE_PREPROCESSING_HEIGHT: int = 256
    IMAGE_PREPROCESSING_WIDTH: int = 256
    IMAGE_PREPROCESSING_DEPTH: int = 3

    EGRESS_PATH: str = os.path.abspath(
        os.path.join(os.path.pardir, "models/kag_dogs_vs_cats"))
    MODEL_PATH: str = os.path.join(EGRESS_PATH, "alexnet_dogs_vs_cats.model")
    MEAN_RGB: str = os.path.join(DATASET_PATH, "dog_vs_cats_mean_rgb.json")
    LABEL_FROM_PATH = lambda image_path: image_path.split("/")[-1].split(".")[0
                                                                              ]


@dataclass(frozen=True)
class AlexNetImage:
    height: int = 227
    width: int = 227
    depth: int = 3
