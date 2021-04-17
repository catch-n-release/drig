from imutils import paths
import progressbar
from drig.feature.condenser import FeatureCondenser
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder
from keras.applications import imagenet_utils
from keras.preprocessing.image import load_img, img_to_array


class FeatureExtractor:
    def __init__(self,
                 image_dataset: str,
                 feature_dataset: str,
                 network,
                 batch_size: int = 32,
                 buffer_size: int = 1000,
                 shuffle: bool = True):
        try:
            self.image_datum_path = image_dataset
            self.feature_datum_path = feature_dataset
            self.batch_size = batch_size
            self.buffer_size = buffer_size
            self.net = network
            self.image_paths = list(paths.list_images(self.image_datum_path))
            self.num_images = len(self.image_paths)
            if shuffle:
                random.shuffle(self.image_paths)

        except Exception as e:
            raise e

    def encode_labels(self):
        try:
            self.labels = [
                image_path.split("/")[-2] for image_path in self.image_paths
            ]
            label_encoder = LabelEncoder()
            self.labels = label_encoder.fit_transform(self.labels)
            self.classes = label_encoder.classes_
        except Exception as e:
            raise e

    def feature_condeser(self,
                         feature_size: int,
                         group_name: str = "features"):
        try:
            feature_datum = FeatureCondenser((self.num_images, feature_size),
                                             self.feature_datum_path,
                                             group_name=group_name,
                                             buffer_size=self.buffer_size)
            feature_datum.save_class_names(self.classes)
            return feature_datum
        except Exception as e:
            raise e

    def extract_features(self,
                         target_image_dim: tuple,
                         feature_size: int,
                         group_name: str = None):
        try:
            self.encode_labels()
            feature_datum = self.feature_condeser(feature_size, group_name)
            widgets = [
                f"Extracting Features: ",
                progressbar.Percentage(),
                " ",
                progressbar.Bar(marker="⎍"),
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
                    image = load_img(image_path, target_size=target_image_dim)
                    image = img_to_array(image)
                    image = np.expand_dims(image, axis=0)
                    image = imagenet_utils.preprocess_input(image)
                    batch_images.append(image)
                batch_images = np.vstack(batch_images)
                batch_features = self.net.predict(batch_images,
                                                  batch_size=self.batch_size)
                batch_features = batch_features.reshape(
                    (batch_features.shape[0], 512))
                feature_datum.commit(batch_features, batch_labels)
                prog_bar.update(index)
            feature_datum.lock()
            prog_bar.finish()
        except Exception as e:
            raise e
