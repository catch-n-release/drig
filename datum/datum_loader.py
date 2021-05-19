import numpy as np
import cv2
import progressbar
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


class ImageDatumLoader:
    def __init__(self, preprocessors=None):
        try:
            self.preprocessors = preprocessors
            if self.preprocessors is None:
                self.preprocessors = []
        except Exception as e:
            raise e

    def load(self,
             image_paths,
             verbose=-1,
             input_dim=None,
             normalize_data=False):
        try:
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
            prog_bar = progressbar.ProgressBar(maxval=len(image_paths),
                                               widgets=widgets).start()

            for i, image_path in enumerate(image_paths):

                image = cv2.imread(image_path)
                label = image_path.split("/")[-2]
                if self.preprocessors:
                    for preprocessor in self.preprocessors:
                        image = preprocessor.preprocess(image)
                data.append(image)
                labels.append(label)

                prog_bar.update(i)
            prog_bar.finish()
            data = np.array(data)
            labels = np.array(labels)
            print(f"Processed {i+1}/{len(image_paths)} Images.")
            if input_dim:
                flattened_data = data.reshape(data.shape[0],
                                              np.prod(np.array(input_dim)))
                print(
                    f"Feature Matrix Memory Size: {flattened_data.nbytes/(1024*1000.0)} MB"
                )
            if normalize_data:
                data = data.astype("float") / 255.0
            return data, labels

        except Exception as e:
            raise e


class CSVDatumLoader:
    def __init__(
        self,
        csv_path: str,
        features: list = None,
        splitter: str = " ",
        skewed_filter: dict = None,
    ):
        try:
            self.csv_path = csv_path
            self.features = features
            self.dataframe = pd.read_csv(
                self.csv_path,
                sep=splitter,
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
                    test_x_cat = np.hstack([test_x_cat, test_encoded_feature])

            train_x = np.hstack([train_x_cat, train_x_cont])
            test_x = np.hstack([test_x_cat, test_x_cont])

            return train_x, test_x
        except Exception as e:
            raise e
