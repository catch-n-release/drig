from dataclasses import dataclass, field
from typing import Dict
import os
from drig.config import TinyVGGNetImage


@dataclass(frozen=True)
class DogsVsCatsConfig:
    DATASET_PATH: str = os.path.abspath(
        os.path.join(
            os.path.pardir,
            "datasets/kag_dogs_vs_cats",
        ))
    TRAINING_IMAGES_PATH: str = os.path.join(
        DATASET_PATH,
        "train",
    )
    CLASS_INDEX: str = -3
    FEATURE_DATUM_PATH: str = os.path.join(
        DATASET_PATH,
        "hdf5_datum",
    )
    TRAINING_DATUM_PATH: str = os.path.join(
        FEATURE_DATUM_PATH,
        "training_datum.hdf5",
    )
    VALIDATION_DATUM_PATH: str = os.path.join(
        FEATURE_DATUM_PATH,
        "validation_datum.hdf5",
    )
    TESTING_DATUM_PATH: str = os.path.join(
        FEATURE_DATUM_PATH,
        "testing_datum.hdf5",
    )
    MEAN_RGB_PATH: str = os.path.join(
        DATASET_PATH,
        "dog_vs_cats_mean_rgb.json",
    )

    NUM_CLASSES: int = 2
    NUM_VALIDATION_IMAGES: int = 1250 * NUM_CLASSES
    NUM_TESTING_IMAGES: int = 1250 * NUM_CLASSES

    IMAGE_PREPROCESSING_HEIGHT: int = 256
    IMAGE_PREPROCESSING_WIDTH: int = 256
    IMAGE_PREPROCESSING_DEPTH: int = 3

    BATCH_SIZE = 64
    ALPHA = 1e-4
    EPOCHS = 75

    EFFLUX_PATH: str = os.path.abspath(
        os.path.join(
            os.path.pardir,
            "models/AlexNet",
        ))
    NET_PATH: str = os.path.join(
        EFFLUX_PATH,
        f"alexnet_dogs_vs_cats_{BATCH_SIZE}_{ALPHA}.model",
    )

    PLOT_PATH: str = os.path.join(
        EFFLUX_PATH,
        f"alexnet_dogs_vs_cats_{BATCH_SIZE}_{ALPHA}.png",
    )
    JSON_PATH: str = os.path.join(
        EFFLUX_PATH,
        f"alexnet_dogs_vs_cats_{BATCH_SIZE}_{ALPHA}.json",
    )


@dataclass(frozen=True)
class TinyImageNetConfig:
    DATASET_PATH: str = os.path.abspath(
        os.path.join(
            os.path.pardir,
            "datasets/tiny-imagenet-200",
        ))

    TRAINING_IMAGES_PATH: str = os.path.join(
        DATASET_PATH,
        "train",
    )
    CLASS_INDEX: str = -4
    VALIDATION_IMAGES_PATH: str = os.path.join(
        DATASET_PATH,
        "val/images",
    )
    VALIDATION_MAPPINGS: str = os.path.join(
        DATASET_PATH,
        "val/val_annotations.txt",
    )
    WORDNET_IDS: str = os.path.join(
        DATASET_PATH,
        "wnids.txt",
    )
    WORDNET_LABELS: str = os.path.join(
        DATASET_PATH,
        "words.txt",
    )
    MEAN_RGB_PATH: str = os.path.join(
        DATASET_PATH,
        "tiny_imagenet_mean_rgb.json",
    )

    FEATURE_DATUM_PATH: str = os.path.join(
        DATASET_PATH,
        "hdf5_datum",
    )
    TRAINING_DATUM_PATH: str = os.path.join(
        FEATURE_DATUM_PATH,
        "training_datum.hdf5",
    )
    VALIDATION_DATUM_PATH: str = os.path.join(
        FEATURE_DATUM_PATH,
        "validation_datum.hdf5",
    )
    TESTING_DATUM_PATH: str = os.path.join(
        FEATURE_DATUM_PATH,
        "testing_datum.hdf5",
    )

    NUM_CLASSES: int = 200
    NUM_TESTING_IMAGES: int = 50 * NUM_CLASSES

    IMAGE_PREPROCESSING_HEIGHT: int = 64
    IMAGE_PREPROCESSING_WIDTH: int = 64
    IMAGE_PREPROCESSING_DEPTH: int = 3

    EFFLUX_PATH: str = os.path.abspath(
        os.path.join(
            os.path.pardir,
            "models/NighGoogeLeNet",
        ))
    NET_PATH: str = os.path.join(
        EFFLUX_PATH,
        "model_tracks/tiny_imagenet_epoch_70.hdf5",
    )
    PLOT_PATH: str = os.path.join(
        EFFLUX_PATH,
        "tiny_imagenet.png",
    )
    JSON_PATH: str = os.path.join(
        EFFLUX_PATH,
        "tiny_imagenet.json",
    )


@dataclass(frozen=True)
class CALTECH101Config:
    DATASET_PATH: str = os.path.abspath(
        os.path.join(
            os.path.pardir,
            "datasets/101_ObjectCategories",
        ))
    CLASS_INDEX: int = -3
    FEATURE_DATASET_DIR_PATH: str = os.path.join(
        os.path.dirname(DATASET_PATH),
        "features/101_ObjectCategories",
    )
    VGG16_FEATURE_DATUM_PATH: str = os.path.join(
        FEATURE_DATASET_DIR_PATH,
        "VGG16_features.hdf5",
    )

    # FEATURE EXTRACTOR CAST
    BATCH_SIZE: int = 64
    BUFFER_SIZE: int = 1600

    # ML NETWORK CAST # LOGISTIC REGRESSION CAST
    TRAIN_SIZE: float = 0.75
    PARAM_MESH: Dict[str, list] = field(default_factory=lambda: (dict(
        C=[0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0])))
    CV: int = 3
    JOBS: int = -1

    # UPSHOTS CAST
    EFFLUX_PATH: str = os.path.abspath(os.path.join(
        os.path.pardir,
        "models",
    ))
    VGG16_NET_PATH: str = os.path.join(
        EFFLUX_PATH,
        "VGG16/CALTECH101_tl.cpickle",
    )


@dataclass(frozen=True)
class AnimalsConfig:
    DATASET_PATH: str = os.path.abspath(
        os.path.join(
            os.path.pardir,
            "datasets/animals",
        ))


@dataclass(frozen=True)
class Flowers17Config:
    DATASET_PATH: str = os.path.abspath(
        os.path.join(
            os.path.pardir,
            "datasets/17flowers",
        ))
    CLASS_INDEX: int = -3

    # PREPROCESSING CAST
    IMAGE_PREPROCESSING_HEIGHT: int = TinyVGGNetImage.HEIGHT
    IMAGE_PREPROCESSING_WIDTH: int = TinyVGGNetImage.WIDTH
    IMAGE_PREPROCESSING_DEPTH: int = TinyVGGNetImage.DEPTH

    # TRAINING CAST
    TEST_SIZE: float = 0.25
    ALPHA: float = 5e-2
    BATCH_SIZE: int = 64
    EPOCHS: int = 100

    # UPSHOTS CAST
    EFFLUX_PATH: str = os.path.abspath(os.path.join(
        os.path.pardir,
        "models",
    ))
    VGG16_NET_PATH: str = os.path.join(
        EFFLUX_PATH,
        "TinyVGGNet/flowers17/flowers17.h5",
    )

    VGG16_LOSS_ACC_PLOT_PATH: str = os.path.join(
        EFFLUX_PATH,
        "TinyVGGNet/flowers17/flowers17_regularized.png",
    )
    JSON_PATH: str = os.path.join(
        EFFLUX_PATH,
        "TinyVGGNet/flowers17/flowers17_regularized.json",
    )

    REGULARIZED_VGG16_NET_PATH: str = os.path.join(
        EFFLUX_PATH, "TinyVGGNet/flowers17/flowers17_regularized.h5")
    REGULARIZED_VGG16_LOSS_ACC_PLOT_PATH: str = os.path.join(
        EFFLUX_PATH,
        "TinyVGGNet/flowers17/flowers17_regularized.png",
    )
    REGULARIZED_JSON_PATH: str = os.path.join(
        EFFLUX_PATH,
        "TinyVGGNet/flowers17/flowers17_regularized.json",
    )


@dataclass(frozen=True)
class CIFAR10Config:
    DATASET_PATH: str = os.path.abspath(
        os.path.join(
            os.path.pardir,
            "datasets/CIFAR-10",
        ))
    CLASSES: int = 10

    # IMAGE CAST

    IMAGE_HEIGHT: int = 32
    IMAGE_WIDTH: int = 32
    IMAGE_DEPTH: int = 3

    # TRAINING CAST
    TEST_SIZE: float = 0.25
    INIT_ALPHA: float = 1e-1
    BATCH_SIZE: int = 64
    L2_REGULATION: float = 1e-4
    EPOCHS: int = 100
    RESNET_CONFIG: Dict[int, tuple] = field(default_factory=lambda: ({
        0: (0, 64),
        1: (9, 64),
        2: (9, 128),
        3: (9, 256),
    }))
    STARTING_EPOCH = 0

    # UPSHOTS CAST
    RESNET_EFFLUX_PATH: str = os.path.abspath(
        os.path.join(
            os.path.pardir,
            "models/ResNet",
        ))

    REFINED_RESNET_PATH: str = os.path.join(
        RESNET_EFFLUX_PATH,
        f"resnet56_cifar10_{BATCH_SIZE}_{INIT_ALPHA}_decay.h5",
    )

    RESNET_TRACKS_PATH: str = os.path.join(
        RESNET_EFFLUX_PATH,
        "network_tracks",
    )

    RESNET_PLOT_PATH: str = os.path.join(
        RESNET_EFFLUX_PATH,
        f"resnet56_cifar10_{BATCH_SIZE}_{INIT_ALPHA}_decay.png",
    )

    RESNET_JSON_PATH: str = os.path.join(
        RESNET_EFFLUX_PATH,
        f"resnet56_cifar10_{BATCH_SIZE}_{INIT_ALPHA}_decay.json",
    )
