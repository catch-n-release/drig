from imutils import paths
import progressbar
from drig.feature import FeatureCondenser
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from drig.utils import log
import os


class FeatureExtractor:
    def __init__(
        self,
        feature_datum_path: str,
        class_index: int,
        network,
        image_datum_path: str = None,
        net_input_cast: tuple = None,
        batch_size: int = 64,
        buffer_size: int = 1600,
        shuffle: bool = True,
        preprocessor: object = None,
        image_net: bool = False,
        image_paths: list = None,
    ):
        try:
            self.image_datum_path = image_datum_path if not image_paths else None
            self.feature_datum_path = feature_datum_path
            self.class_index = class_index
            self.batch_size = batch_size
            self.buffer_size = buffer_size
            self.net = network
            self.net_input_cast = net_input_cast
            self.image_paths = list(paths.list_images(self.image_datum_path)
                                    ) if self.image_datum_path else image_paths

            self.num_images = len(self.image_paths)
            self.preprocessor = preprocessor
            self.image_net = image_net
            if shuffle:
                np.random.shuffle(self.image_paths)
            self.feature_size = FeatureExtractor.unit_image_feature(
                self.net,
                np.random.choice(self.image_paths),
                self.net_input_cast,
                self.preprocessor,
                self.image_net,
            ).shape[1]
            os.makedirs(os.path.dirname(self.feature_datum_path),
                        exist_ok=True)
        except Exception as e:
            raise e

    def encode_labels(self):
        try:
            log.info("ENCODING LABELS")
            self.labels = [
                image_path.replace(".", " ").replace(
                    "/", " ").split(" ")[self.class_index]
                for image_path in self.image_paths
            ]
            label_encoder = LabelEncoder()
            self.encoded_labels = label_encoder.fit_transform(self.labels)
            self.classes = label_encoder.classes_
            log.info(f"CLASSES : {len(self.classes)}")
        except Exception as e:
            raise e

    def feature_condenser(
        self,
        group_name: str = "features",
    ):
        try:
            log.info("INITIALZING FEATURE CONDENSER")
            feature_datum = FeatureCondenser(
                (self.num_images, self.feature_size),
                self.feature_datum_path,
                group_name=group_name,
                buffer_size=self.buffer_size)
            feature_datum.save_class_names(self.classes)
            return feature_datum
        except Exception as e:
            raise e

    def extract_features(
        self,
        group_name: str = "features",
    ):
        try:
            self.encode_labels()
            feature_datum = self.feature_condenser(group_name)
            widgets = [
                "Extracting Features: ",
                progressbar.Percentage(),
                " ",
                progressbar.Bar(marker="❆"),
                " ",
                progressbar.SimpleProgress(),
                " ",
                progressbar.ETA(),
            ]
            prog_bar = progressbar.ProgressBar(maxval=self.num_images,
                                               widgets=widgets).start()
            for index in np.arange(0, self.num_images, self.batch_size):
                batch_image_paths = self.image_paths[index:index +
                                                     self.batch_size]
                batch_labels = self.encoded_labels[index:index +
                                                   self.batch_size]
                batch_images = list()
                for image_path in batch_image_paths:
                    image_for_prediction = FeatureExtractor.preprocess_image(
                        image_path,
                        self.net_input_cast,
                        self.preprocessor,
                        self.image_net,
                    )
                    batch_images.append(image_for_prediction)

                batch_images = np.vstack(batch_images)

                batch_features = self.net.predict(batch_images,
                                                  batch_size=len(batch_images))
                batch_features = batch_features.reshape(
                    (batch_features.shape[0], self.feature_size))

                feature_datum.commit(batch_features, batch_labels)
                prog_bar.update(index)
            feature_datum.seal()
            prog_bar.finish()
        except Exception as e:
            raise e

    @staticmethod
    def preprocess_image(
        image_path,
        net_input_cast: tuple = None,
        preprocessor: object = None,
        image_net: bool = False,
    ):
        try:
            image = load_img(image_path, target_size=net_input_cast)
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
            if preprocessor:
                image = preprocessor.preprocess_input(image)
            elif image_net:
                image = imagenet_utils.preprocess_input(image)
            return image
        except Exception as e:
            raise e

    @staticmethod
    def unit_image_feature(
        net,
        image_path: str,
        net_input_cast: tuple = None,
        preprocessor: object = None,
        image_net: bool = False,
    ):
        try:
            image_for_prediction = FeatureExtractor.preprocess_image(
                image_path, net_input_cast, preprocessor, image_net)
            raw_feature_vector = net.predict(image_for_prediction)
            feature_vector_size = np.prod(raw_feature_vector.shape[1:])
            feature_vector = raw_feature_vector.reshape(
                raw_feature_vector.shape[0], feature_vector_size)

            return feature_vector
        except Exception as e:
            raise e
