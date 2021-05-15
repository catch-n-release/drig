from imutils import paths
import progressbar
from drig.feature import FeatureCondenser
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.applications import imagenet_utils
from keras.preprocessing.image import load_img, img_to_array
import os


class FeatureExtractor:
    def __init__(self,
                 feature_datum_path: str,
                 label_index: int,
                 network,
                 image_datum_path: str = None,
                 net_input_dim: tuple = (224, 224),
                 batch_size: int = 64,
                 buffer_size: int = 1000,
                 shuffle: bool = True,
                 image_net: bool = True,
                 image_paths: list = None):
        try:
            self.image_datum_path = image_datum_path if not image_paths else None
            self.feature_datum_path = feature_datum_path
            self.label_index = label_index
            self.feature_dir = str("/").join(
                self.feature_datum_path.split("/")[:-1])
            if not os.path.exists(self.feature_dir):
                os.makedirs(self.feature_dir)
            self.batch_size = batch_size
            self.buffer_size = buffer_size
            self.net = network
            self.net_input_dim = net_input_dim
            self.image_paths = list(paths.list_images(self.image_datum_path)
                                    ) if self.image_datum_path else image_paths

            self.num_images = len(self.image_paths)
            self.image_net = image_net
            if shuffle:
                np.random.shuffle(self.image_paths)
            self.feature_size = FeatureExtractor.feature_size(
                self.net, np.random.choice(self.image_paths),
                self.net_input_dim, self.image_net)

        except Exception as e:
            raise e

    def encode_labels(self):
        try:
            self.labels = [
                image_path.replace(".", " ").replace(
                    "/", " ").split(" ")[self.label_index]
                for image_path in self.image_paths
            ]
            label_encoder = LabelEncoder()
            self.labels = label_encoder.fit_transform(self.labels)
            self.classes = label_encoder.classes_
        except Exception as e:
            raise e

    def condeser_features(self, group_name: str):
        try:
            feature_datum = FeatureCondenser(
                (self.num_images, self.feature_size),
                self.feature_datum_path,
                group_name=group_name,
                buffer_size=self.buffer_size)
            feature_datum.save_class_names(self.classes)
            return feature_datum
        except Exception as e:
            raise e

    def extract_features(self, group_name: str = "features"):
        try:
            self.encode_labels()
            feature_datum = self.condeser_features(group_name)
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
                batch_labels = self.labels[index:index + self.batch_size]
                batch_images = list()
                for image_path in batch_image_paths:
                    image_for_prediction = FeatureExtractor.preprocess_image(
                        image_path, self.net_input_dim, self.image_net)
                    batch_images.append(image_for_prediction)

                batch_images = np.vstack(batch_images)

                batch_features = self.net.predict(batch_images,
                                                  batch_size=len(batch_images))
                batch_features = batch_features.reshape(
                    (batch_features.shape[0], self.feature_size))

                feature_datum.commit(batch_features, batch_labels)
                prog_bar.update(index)
            feature_datum.lock()
            prog_bar.finish()
        except Exception as e:
            raise e

    @staticmethod
    def preprocess_image(image_path, net_input_dim, image_net: bool = True):
        try:
            image = load_img(image_path, target_size=net_input_dim)
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
            if image_net:
                image = imagenet_utils.preprocess_input(image)
            return image
        except Exception as e:
            raise e

    @staticmethod
    def feature_size(net,
                     image_path: str,
                     net_input_dim: tuple,
                     image_net: bool = True):
        try:
            image_for_prediction = FeatureExtractor.preprocess_image(
                image_path, net_input_dim, image_net)
            feature_vector = net.predict(image_for_prediction)
            feature_vector_size = np.prod(feature_vector.shape[1:])
            return feature_vector_size
        except Exception as e:
            raise e
