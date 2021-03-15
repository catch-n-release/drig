import numpy as np
import cv2


class DatasetLoader:
    def __init__(self, preprocessors=None):
        try:
            self.preprocessors = preprocessors
            if self.preprocessors is None:
                self.preprocessors = []
        except Exception as e:
            raise e

    def load(self, image_paths, verbose=-1):
        try:
            data = list()
            labels = list()

            for i, image_path in enumerate(image_paths):

                image = cv2.imread(image_path)
                label = image_path.split("/")[-2]
                if self.preprocessors:
                    for preprocessor in self.preprocessors:
                        preprocessor.preprocess(image)
                data.append(image)
                labels.append(label)

            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print(f"Processed {i+1}/{len(image_paths)} Images.")

            return (np.array(data), np.array(labels))

        except Exception as e:
            raise e
