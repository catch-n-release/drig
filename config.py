import enum
import os
import logging as log

log.basicConfig(level=log.INFO)


class DogsVsCatsConfig(enum.Enum):
    DATASET_PATH = os.path.abspath(
        os.path.join(os.path.pardir, "datasets/kag_dogs_vs_cats"))
    IMAGES_PATH = os.path.join(DATASET_PATH, "train")
    HDF5_DATUM_PATH = os.path.join(DATASET_PATH, "hdf5_datum")
    TRAINING_DATUM_PATH = os.path.join(HDF5_DATUM_PATH, "training_datum.hdf5")
    VALIDATION_DATUM_PATH = os.path.join(HDF5_DATUM_PATH,
                                         "validation_datum.hdf5")
    TESTING_DATUM_PATH = os.path.join(HDF5_DATUM_PATH, "testing_datum.hdf5")

    NUM_CLASSES = 2
    NUM_VALIDATION_IMAGES = 1250 * NUM_CLASSES
    NUM_TESTING_IMAGES = 1250 * NUM_CLASSES

    EGRESS_PATH = os.path.abspath(
        os.path.join(os.path.pardir, "models/kag_dogs_vs_cats"))
    MODEL_PATH = os.path.join(EGRESS_PATH, "alexnet_dogs_vs_cats.model")
    MEAN_RGB = os.path.join(DATASET_PATH, "dog_vs_cats_mean_rgb.json")
    LABEL_FROM_PATH = lambda image_path: image_path.split("/")[-1].split(".")[0
                                                                              ]
