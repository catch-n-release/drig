from dataclasses import dataclass, field
from typing import Dict
import os


@dataclass(frozen=True)
class DogsVsCatsConfig:
    DATASET_PATH: str = os.path.abspath(
        os.path.join(os.path.pardir, "datasets/kag_dogs_vs_cats"))
    TRAINING_IMAGES_PATH: str = os.path.join(DATASET_PATH, "train")
    LABEL_INDEX: str = -3
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
    NET_PATH: str = os.path.join(
        EGRESS_PATH, f"alexnet_dogs_vs_cats_{BATCH_SIZE}_{ALPHA}.model")

    PLOT_PATH: str = os.path.join(
        EGRESS_PATH, f"alexnet_dogs_vs_cats_{BATCH_SIZE}_{ALPHA}.png")
    JSON_PATH: str = os.path.join(
        EGRESS_PATH, f"alexnet_dogs_vs_cats_{BATCH_SIZE}_{ALPHA}.json")


@dataclass(frozen=True)
class TinyImageNetConfig:
    DATASET_PATH: str = os.path.abspath(
        os.path.join(os.path.pardir, "datasets/tiny-imagenet-200"))

    TRAINING_IMAGES_PATH: str = os.path.join(DATASET_PATH, "train")
    LABEL_INDEX: str = -4
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

    EGRESS_PATH: str = os.path.abspath(
        os.path.join(os.path.pardir, "models/NighGoogeLeNet"))
    NET_PATH: str = os.path.join(EGRESS_PATH,
                                 "model_tracks/tiny_imagenet_epoch_70.hdf5")
    PLOT_PATH: str = os.path.join(EGRESS_PATH, "tiny_imagenet.png")
    JSON_PATH: str = os.path.join(EGRESS_PATH, "tiny_imagenet.json")


@dataclass(frozen=True)
class CALTECH101Config:
    DATASET_PATH: str = os.path.abspath(
        os.path.join(os.path.pardir, "datasets/101_ObjectCategories"))
    LABEL_INDEX: int = -3
    FEATURE_DATASET_DIR_PATH: str = os.path.join(
        os.path.dirname(DATASET_PATH), "features/101_ObjectCategories")
    VGG16_FEATURE_DATUM_PATH: str = os.path.join(FEATURE_DATASET_DIR_PATH,
                                                 "VGG16_features.hdf5")

    # FEATURE EXTRACTOR CAST
    BATCH_SIZE: int = 64
    BUFFER_SIZE: int = 1600

    # ML NETWORK CAST # LOGISTIC REGRESSION CAST
    TRAIN_SIZE: float = 0.75
    PARAM_MESH: Dict[str, list] = field(default_factory=lambda: (dict(
        C=[0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0])))
    CV: int = 3
    JOBS: int = -1

    EGRESS_PATH: str = os.path.abspath(os.path.join(os.path.pardir, "models"))
    VGG16_NET_PATH: str = os.path.join(EGRESS_PATH,
                                       "VGG16/CALTECH101_tl.cpickle")
