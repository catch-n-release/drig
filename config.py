from dataclasses import dataclass
import os
import logging

logging.basicConfig(level=logging.INFO)


@dataclass(frozen=True)
class DogsVsCats:
    DATASET_PATH: str = os.path.abspath(
        os.path.join(os.path.pardir, "datasets/kag_dogs_vs_cats"))
    TRAINING_IMAGES_PATH: str = os.path.join(DATASET_PATH, "train")

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


@dataclass(frozen=True)
class TinyImageNet:
    DATASET_PATH: str = "/Users/suyashsrivastava/drig/datasets/tiny-imagenet-200"

    TRAINING_IMAGES_PATH: str = os.path.join(DATASET_PATH, "train")
    TRAINING_LABEL_INDEX: str = -4
    VALIDATION_IMAGES_PATH: str = os.path.join(DATASET_PATH, "val/images")
    VALIDATION_MAPPINGS: str = os.path.join(DATASET_PATH,
                                            "val/val_annotations.txt")
    WORDNET_IDS: str = os.path.join(DATASET_PATH, "wnids.txt")
    WORDNET_LABELS: str = os.path.join(DATASET_PATH, "words.txt")
    MEAN_RGB_PATH: str = os.path.join(DATASET_PATH,
                                      "tiny_imagenet_mean_rgb.json")

    HDF5_DATUM_PATH: str = os.path.join(DATASET_PATH, "hdf5_datum")
    TRAINING_DATUM_PATH: str = os.path.join(HDF5_DATUM_PATH,
                                            "training_datum.hdf5")
    VALIDATION_DATUM_PATH: str = os.path.join(HDF5_DATUM_PATH,
                                              "validation_datum.hdf5")
    TESTING_DATUM_PATH: str = os.path.join(HDF5_DATUM_PATH,
                                           "testing_datum.hdf5")

    NUM_CLASSES: int = 200
    NUM_TESTING_IMAGES: int = 50 * NUM_CLASSES

    IMAGE_PREPROCESSING_HEIGHT: int = 64
    IMAGE_PREPROCESSING_WIDTH: int = 64
    IMAGE_PREPROCESSING_DEPTH: int = 3

    EGRESS_PATH: str = "/Users/suyashsrivastava/drig/models/GoogeLeNet"
    MODEL_PATH: str = os.path.join(EGRESS_PATH,
                                   "checkpoints/tiny_imagenet_epoch_70.hdf5")

    PLOT_PATH: str = os.path.join(EGRESS_PATH, "tiny_imagenet.png")
    JSON_PATH: str = os.path.join(EGRESS_PATH, "tiny_imagenet.json")
