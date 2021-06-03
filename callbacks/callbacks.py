from keras.callbacks import BaseLogger
import os
import json
from drig.utils import plot_training_metrics
from drig.utils import log
from keras.callbacks import Callback


class LossAccuracyTracker(BaseLogger):
    """docstring for LossAccuracyTracker"""
    def __init__(self,
                 plot_image_path: str,
                 json_path: str = None,
                 starting_epoch: int = 0):

        super(LossAccuracyTracker, self).__init__()

        self.plot_image_path = plot_image_path
        self.json_path = json_path
        os.makedirs(os.path.dirname(self.plot_image_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.json_path), exist_ok=True)
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
                with open(self.json_path, "w") as json_file:
                    json_file.write(json.dumps(self.net_training_history))

            if len(self.net_training_history["loss"]) > 1:
                plot_training_metrics(
                    epochs=epoch,
                    model_training_history=self.net_training_history,
                    save_path=self.plot_image_path,
                    callback=True,
                )

        except Exception as e:
            raise e


class AlphaScheduler:
    def __init__(self, base_alpha, total_epochs):
        try:
            self.base_alpha = base_alpha
            self.total_epochs = total_epochs
        except Exception as e:
            raise e

    def polynomial_decay(self, epoch):
        try:

            exp = 1.0  # Linear

            new_alpha = self.base_alpha * (
                1 - (epoch / float(self.total_epochs)))**exp
            log.info(f"---SETTING NEW ALPHA : {new_alpha}")
            return new_alpha

        except Exception as e:
            raise e


class NetworkTracker(Callback):
    def __init__(self, model_dir, epoch_interval=5, starting_epoch=0):

        super(Callback, self).__init__()

        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.epoch_interval = epoch_interval
        self.epoch_head = starting_epoch

    def on_epoch_end(self, epoch, logs={}):
        self.epoch_head += 1

        if self.epoch_head % self.epoch_interval == 0:
            model_save_path = os.path.join(self.model_dir,
                                           f"epoch_{self.epoch_head}.hdf5")
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            self.model.save(model_save_path, overwrite=True)
