import os
import h5py


class FeatureCondenser:
    def __init__(self,
                 shape,
                 output_model_path,
                 group_name="images",
                 buffer_size=1000):
        try:
            if os.path.exists(output_model_path):
                raise Exception("File Already Exists!")
            self.global_index = 0
            self.buffer_size = buffer_size
            self.buffer = dict(data=list(), labels=list())
            self.datum = h5py.File(output_model_path, "w")
            self.data = self.datum.create_dataset(group_name,
                                                  shape,
                                                  dtype="float")
            self.labels = self.datum.create_dataset("labels", (shape[0], ),
                                                    dtype="int")
        except Exception as e:
            raise e

    def commit(self, rows, labels):
        try:
            self.buffer.get("data").extend(rows)
            self.buffer.get("labels").extend(labels)
            if len(self.buffer.get("data")) >= self.buffer_size:
                self.push()
        except Exception as e:
            raise e

    def push(self):
        try:
            buffer_data = self.buffer.get("data")
            local_index = self.global_index + len(buffer_data)
            self.data[self.global_index:local_index] = buffer_data
            self.labels[self.global_index:local_index] = self.buffer.get(
                "labels")
            self.global_index = local_index
            self.buffer = dict(data=list(), labels=list())
        except Exception as e:
            raise e

    def save_class_names(self, class_names):
        try:
            class_labels = self.datum.create_dataset(
                "class_labels", (len(class_names), ),
                dtype=h5py.special_dtype(vlen=str))
            class_labels[:] = class_names
        except Exception as e:
            raise e

    def latch(self):
        try:
            if len(self.buffer.get("data")) > 0:
                self.push()
            self.datum.close()
        except Exception as e:
            raise e
