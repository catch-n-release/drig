import numpy as np
import cv2
import progressbar


class DatumLoader:
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
                # if verbose > 0 and i >= 0 and (i + 1) % verbose == 0:

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
