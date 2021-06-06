import numpy as np
import cv2
import progressbar
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from drig.utils import log, grab_image_paths
from drig.config import DataType
import glob
import csv


class ImageDatumLoader:
    def __init__(
        self,
        preprocessors: list = None,
    ):
        try:
            self.preprocessors = preprocessors
            if self.preprocessors is None:
                self.preprocessors = []
        except Exception as e:
            raise e

    def load(
        self,
        dataset_path: str,
        class_index: int,
        input_cast: tuple = None,
        normalized: bool = True,
    ):
        try:
            image_paths = grab_image_paths(dataset_path)
            data = list()
            labels = list()

            widgets = [
                "Loading Images :",
                progressbar.Percentage(),
                " ",
                progressbar.Bar(marker="❆"),
                " ",
                progressbar.SimpleProgress(),
                " ",
                progressbar.ETA(),
            ]
            prog_bar = progressbar.ProgressBar(
                maxval=len(image_paths),
                widgets=widgets,
            ).start()

            for i, image_path in enumerate(image_paths):

                image = cv2.imread(image_path)
                label = image_path.replace(".", " ").replace(
                    "/", " ").split(" ")[class_index]

                if self.preprocessors:
                    for preprocessor in self.preprocessors:
                        image = preprocessor.preprocess(image)
                data.append(image)
                labels.append(label)

                prog_bar.update(i)
            self.classes = np.unique(labels)
            prog_bar.finish()
            data = np.array(data)
            labels = np.array(labels)
            log.info(f"PROCESSED {i+1}/{len(image_paths)} IMAGES.")
            if input_cast:
                flattened_data = data.reshape(
                    data.shape[0],
                    np.prod(np.array(input_cast)),
                )
                log.info(
                    f"MEMORY SIZE OCCUPIED: {flattened_data.nbytes/(1024*1000.0)} MB"
                )
            if normalized:
                data = data.astype("float") / 255.0
            return data, labels

        except Exception as e:
            raise e


class CSVDatumLoader:
    def __init__(
        self,
        csv_dir_path: str,
        features: list = None,
        splitter: str = " ",
        skewed_filter: dict = None,
    ):
        try:

            self.csv_file_path = glob.glob(f"{csv_dir_path}/*.csv")[0]
            self.features = features
            self.splitter = splitter
            self.dataframe = pd.read_csv(
                self.csv_file_path,
                sep=self.splitter,
                header=None,
                names=self.features,
            )
            if skewed_filter:
                for key, value in skewed_filter.items():
                    unique_skewed_column_data = self.dataframe[
                        key].value_counts().keys().tolist()
                    total_count = self.dataframe[key].value_counts().tolist()
                    for (data, count) in zip(unique_skewed_column_data,
                                             total_count):
                        if count < value:
                            indices = self.dataframe[self.dataframe[key] ==
                                                     data].index
                            self.dataframe.drop(indices, inplace=True)

        except Exception as e:
            raise e

    def load(
        self,
        train_split,
        test_split,
        continous_features: list = None,
        categorical_features: list = None,
    ):
        try:
            min_max_scaler = MinMaxScaler()
            train_x_cont = min_max_scaler.fit_transform(
                train_split[continous_features])
            test_x_cont = min_max_scaler.transform(
                test_split[continous_features])

            train_x_cat = np.array(list())
            test_x_cat = np.array(list())
            for discrete_feature in categorical_features:
                label_binarizer = LabelBinarizer().fit(
                    self.dataframe[discrete_feature])
                train_encoded_feature = label_binarizer.transform(
                    train_split[discrete_feature])
                test_encoded_feature = label_binarizer.transform(
                    test_split[discrete_feature])
                if not train_x_cat.any():
                    train_x_cat = train_encoded_feature
                    test_x_cat = test_encoded_feature
                else:
                    train_x_cat = np.hstack(
                        [train_x_cat, train_encoded_feature])
                    test_x_cat = np.hstack([
                        test_x_cat,
                        test_encoded_feature,
                    ])

            train_x = np.hstack([
                train_x_cat,
                train_x_cont,
            ])
            test_x = np.hstack([
                test_x_cat,
                test_x_cont,
            ])

            return train_x, test_x
        except Exception as e:
            raise e

    def load_image_csv(
        self,
        image_cast: tuple,
        class_column_last: bool = True,
        encode_classes: bool = False,
    ):
        try:
            data = list()
            classes = list()
            class_column_index = -1 if class_column_last else 0
            with open(
                    self.csv_file_path,
                    mode="r",
            ) as csv_file:
                csv_data = csv.reader(
                    csv_file,
                    delimiter=self.splitter,
                )
                for row in csv_data:
                    classes.append(row[class_column_index])
                    pels = row[:-1] if class_column_last else row[1:]
                    image_pels = np.array(
                        [int(pel) for pel in pels],
                        dtype="uint8",
                    )
                    image = image_pels.reshape(image_cast)
                    data.append(image)
            if encode_classes:
                classes = LabelEncoder().fit_transform(classes)
            return (
                np.array(data, dtype=DataType.FLOAT32),
                np.array(classes, dtype=DataType.INT),
            )

        except Exception as e:
            raise e
