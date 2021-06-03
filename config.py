from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)


@dataclass(frozen=True)
class AlexNetImage:
    HEIGHT: int = 227
    WIDTH: int = 227
    DEPTH: int = 3


@dataclass(frozen=True)
class VGGNetImage:
    HEIGHT: int = 224
    WIDTH: int = 224
    DEPTH: int = 3


@dataclass(frozen=True)
class TinyVGGNetImage:
    HEIGHT: int = 64
    WIDTH: int = 64
    DEPTH: int = 3


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
    MESH_8x8: tuple = (8, 8)


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


@dataclass(frozen=True)
class ImageCast:
    RGB_32x32: tuple = (32, 32, 3)
    RGB_64x64: tuple = (64, 64, 3)
    RGB_224x224: tuple = (224, 224, 3)
    RGB_227x227: tuple = (227, 227, 3)
    RGB_256x256: tuple = (256, 256, 3)
    GRAY_32x32: tuple = (32, 32, 1)
    GRAY_64x64: tuple = (64, 64, 1)
    GRAY_224x224: tuple = (224, 224, 1)
    GRAY_227x227: tuple = (227, 227, 1)
    GRAY_256x256: tuple = (256, 256, 1)


@dataclass(frozen=True)
class Error:
    DATASET_PATH_ERROR: str = "PROVIDED DATASET PATH IS INVALID."
    IMAGE_PATH_ERROR: str = "PROVIDED IMAGE PATH IS INVALID."
    TRAINING_METRICS_PLOT_ERROR: str = "EITHER JSON PATH OR EPOCHS & MODEL TRAINING HISTORY SHOULD BE SUPPLIED"
    NO_IMAGE_OR_PATH_ERROR: str = "PLEASE PROVIDE EITHER IMAGE OR IMAGE PATH"
    EMPTY_DATASET_ERROR: str = "DIRECTORY HAS NO IMAGES"
    NO_DATASET_OR_IMAGE_PATHS_ERROR: str = "PLEASE PROVIDE EITHER DATASET PATH OR ALL IMAGE PATHS"
    INVALID_PARAM_ERROR: str = "PLEASE PROVIDE VALID PARAMETER(S)"


@dataclass(frozen=True)
class ImageFontPath:
    SKIA: str = "/System/Library/Fonts/Supplemental/Skia.ttf"


@dataclass(frozen=True)
class Loss:
    CAT_CROSS: str = "categorical_crossentropy"
    BI_CROSS: str = "binary_crossentropy"


@dataclass(frozen=True)
class Metrics:
    ACCURACY: str = "accuracy"


@dataclass(frozen=True)
class DataType:
    FLOAT32: str = "float32"
    UINT8: str = "uint8"
