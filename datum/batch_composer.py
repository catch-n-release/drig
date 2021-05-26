import numpy as np
from keras.utils import np_utils
import h5py
from keras.preprocessing.image import ImageDataGenerator


class BatchComposer:
    def __init__(self,
                 datum_path: str,
                 batch_size: int,
                 preprocessors: list = None,
                 data_augmenter: ImageDataGenerator = None,
                 binarize: bool = True,
                 classes: int = 2):
        try:
            self.datum_path = datum_path
            self.batch_size = batch_size
            self.preprocessors = preprocessors
            self.data_augmenter = data_augmenter
            self.binarize = binarize
            self.classes = classes

            self.datum = h5py.File(self.datum_path, mode="r")
            self.total_images = self.datum["labels"].shape[0]
        except Exception as e:
            raise e

    def compose(self, scans: int = np.inf):
        try:
            epochs = 0
            while epochs < scans:
                for index in np.arange(0, self.total_images, self.batch_size):
                    images = self.datum["images"][index:index +
                                                  self.batch_size]
                    labels = self.datum["labels"][index:index +
                                                  self.batch_size]

                    if self.binarize:
                        labels = np_utils.to_categorical(labels, self.classes)

                    if self.preprocessors:
                        preprocessed_images = []

                        for image in images:
                            for preprocessor in self.preprocessors:
                                image = preprocessor.preprocess(image)
                            preprocessed_images.append(image)

                        if len(images) != len(preprocessed_images):
                            raise Exception("Error post image processing")

                        images = np.array(preprocessed_images)

                    if self.data_augmenter:
                        (images, labels) = next(
                            self.data_augmenter.flow(
                                images, labels, batch_size=self.batch_size))

                    yield (images, labels)
                epochs += 1

        except Exception as e:
            raise e

    def halt(self):
        try:
            self.datum.close()
        except Exception as e:
            raise e
