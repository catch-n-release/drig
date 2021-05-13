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
                json_file.close()

            if len(self.net_training_history["loss"]) > 1:
                plot_training_metrics(
                    epochs=epoch,
                    model_training_history=self.net_training_history,
                    save_path=self.plot_image_path,
                    callback=True,
                )

        except Exception as e:
            raise e


class AlphaSchedulers:
    def __init__(self, base_alpha, total_epochs):
        try:
            self.base_alpha = base_alpha
            self.total_epochs = total_epochs
        except Exception as e:
            raise e

    def polynomial_decay(self, epoch):
        try:

            exp = 1.0

            new_alpha = self.base_alpha * (
                1 - (epoch / float(self.total_epochs)))**exp
            log.info(f"---SETTING NEW ALPHA : {new_alpha}")
            return new_alpha

        except Exception as e:
            raise e


# class EpochTracker(Callback):
#     def __init__(self, outputPath, every=5, startAt=0):
#         # call the parent constructor
#         super(Callback, self).__init__()

#         # store the base output path for the model, the number of
#         # epochs that must pass before the model is serialized to
#         # disk and the current epoch value
#         self.outputPath = outputPath
#         self.every = every
#         self.intEpoch = startAt

#     def on_epoch_end(self, epoch, logs={}):
#         # check to see if the model should be serialized to disk
#         if (self.intEpoch + 1) % self.every == 0:
#             p = os.path.sep.join(
#                 [self.outputPath, "epoch_{}.hdf5".format(self.intEpoch + 1)])
#             self.model.save(p, overwrite=True)

#         # increment the internal epoch counter
#         self.intEpoch += 1


class ModelTracker(Callback):
    def __init__(self, model_dir, epoch_interval=5, starting_epoch=0):

        super(Callback, self).__init__()

        self.model_dir = model_dir
        self.epoch_interval = epoch_interval
        self.epoch_head = starting_epoch

    def on_epoch_end(self, epoch, logs={}):
        self.epoch_head += 1

        if self.epoch_head % self.epoch_interval == 0:
            model_save_path = os.path.join(self.model_dir,
                                           f"epoch_{self.epoch_head}.hdf5")
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            self.model.save(model_save_path, overwrite=True)
