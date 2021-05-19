from dataclasses import dataclass
import os
import logging

logging.basicConfig(level=logging.INFO)


@dataclass(frozen=True)
class DogsVsCatsConfig:
    DATASET_PATH: str = os.path.abspath(
        os.path.join(os.path.pardir, "datasets/kag_dogs_vs_cats"))
    TRAINING_IMAGES_PATH: str = os.path.join(DATASET_PATH, "train")
    TRAINING_LABEL_INDEX: str = -3
    HDF5_DATUM_PATH: str = os.path.join(DATASET_PATH, "hdf5_datum")
    TRAINING_DATUM_PATH: str = os.path.join(HDF5_DATUM_PATH,
                                            "training_datum.hdf5")
    VALIDATION_DATUM_PATH: str = os.path.join(HDF5_DATUM_PATH,
                                              "validation_datum.hdf5")
    TESTING_DATUM_PATH: str = os.path.join(HDF5_DATUM_PATH,
                                           "testing_datum.hdf5")
    MEAN_RGB_PATH: str = os.path.join(DATASET_PATH,
                                      "dog_vs_cats_mean_rgb.json")

    NUM_CLASSES: int = 2
    NUM_VALIDATION_IMAGES: int = 1250 * NUM_CLASSES
    NUM_TESTING_IMAGES: int = 1250 * NUM_CLASSES

    IMAGE_PREPROCESSING_HEIGHT: int = 256
    IMAGE_PREPROCESSING_WIDTH: int = 256
    IMAGE_PREPROCESSING_DEPTH: int = 3

    BATCH_SIZE = 64
    ALPHA = 1e-4
    EPOCHS = 75

    EGRESS_PATH: str = os.path.abspath(
        os.path.join(os.path.pardir, "models/AlexNet"))
    MODEL_PATH: str = os.path.join(
        EGRESS_PATH, f"alexnet_dogs_vs_cats_{BATCH_SIZE}_{ALPHA}.model")

    PLOT_PATH: str = os.path.join(
        EGRESS_PATH, f"alexnet_dogs_vs_cats_{BATCH_SIZE}_{ALPHA}.png")
    JSON_PATH: str = os.path.join(
        EGRESS_PATH, f"alexnet_dogs_vs_cats_{BATCH_SIZE}_{ALPHA}.json")


@dataclass(frozen=True)
class AlexNetImage:
    height: int = 227
    width: int = 227
    depth: int = 3


@dataclass(frozen=True)
class TinyImageNetConfig:
    DATASET_PATH: str = os.path.abspath(
        os.path.join(os.path.pardir, "datasets/tiny-imagenet-200"))

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

    EGRESS_PATH: str = "/Users/suyashsrivastava/drig/models/NighGoogeLeNet"
    MODEL_PATH: str = os.path.join(EGRESS_PATH,
                                   "model_tracks/tiny_imagenet_epoch_70.hdf5")
    PLOT_PATH: str = os.path.join(EGRESS_PATH, "tiny_imagenet.png")
    JSON_PATH: str = os.path.join(EGRESS_PATH, "tiny_imagenet.json")


@dataclass(frozen=True)
class Kernel:
    MESH_1x1: tuple = (1, 1)
    MESH_3x3: tuple = (3, 3)
    MESH_5x5: tuple = (5, 5)
    MESH_11x11: tuple = (11, 11)


@dataclass(frozen=True)
class PoolSize:
    MESH_1x1: tuple = (1, 1)
    MESH_2x2: tuple = (2, 2)
    MESH_3x3: tuple = (3, 3)
    MESH_4x4: tuple = (4, 4)
    MESH_5x5: tuple = (5, 5)


@dataclass(frozen=True)
class Stride:
    MESH_1x1: tuple = (1, 1)
    MESH_2x2: tuple = (2, 2)
    MESH_4x4: tuple = (4, 4)


@dataclass(frozen=True)
class Padding:
    SAME: str = "same"
    VALID: str = "valid"


@dataclass(frozen=True)
class Trigger:
    RELU: str = "relu"
    SOFTMAX: str = "softmax"
    LINEAR: str = "linear"
