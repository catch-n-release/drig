from keras.callbacks import BaseLogger
import os
import json
from drig.utils import plot_training_metrics


class LossAccuracyTracker(BaseLogger):
    """docstring for LossAccuracyTracker"""
    def __init__(self,
                 plot_image_path: str,
                 json_path: str = None,
                 starting_epoch: int = 0):
        super(LossAccuracyTracker, self).__init__()
        self.plot_image_path = plot_image_path
        self.json_path = json_path
        self.starting_epoch = starting_epoch

    def on_train_begin(self, logs=dict()):
        try:
            self.net_training_history = dict()
            if self.json_path:
                if os.path.exists(self.json_path):
                    self.net_training_history = json.loads(
                        open(self.json_path).read())

                if self.starting_epoch > 0:
                    for key in self.net_training_history.keys():
                        self.net_training_history[
                            key] = self.net_training_history[
                                key][:self.starting_epoch]

        except Exception as e:
            raise e

    def on_epoch_end(self, epoch, logs=dict()):
        try:

            for key, value in logs.items():
                loss_accuracy_values = self.net_training_history.get(
                    key, list())
                loss_accuracy_values.append(value)
                self.net_training_history[key] = loss_accuracy_values
            if self.json_path:
                json_file = open(self.json_path, "w")
                json_file.write(json.dumps(self.net_training_history))
                json_file.close

            if len(self.net_training_history["loss"]) > 1:
                plot_training_metrics(epoch,
                                      self.net_training_history,
                                      save_path=self.plot_image_path,
                                      callback=True)

        except Exception as e:
            raise e
