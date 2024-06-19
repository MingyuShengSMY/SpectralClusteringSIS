import os

import numpy as np


class TextLog:
    def __init__(self, log_dir, log_file_name):
        self.log_path = os.path.join(log_dir, log_file_name + ".txt")
        with open(self.log_path, "w") as f:
            pass

    def log_a(self, string):
        with open(self.log_path, "a") as f:
            f.write(string)

    def log_w(self, string):
        with open(self.log_path, "w") as f:
            f.write(string)


class CsvLog:
    def __init__(self, log_dir, log_file_name, header: list):

        self.log_path = os.path.join(log_dir, log_file_name + ".csv")

        self.header = header
        self.header = np.array(self.header)
        self.header = self.header.flatten().astype(str)
        self.column_num = len(self.header)

        header = {k: k for k in header}

        self.log_w(header)

    def dict2text(self, array: dict):
        v_list = []
        for k in self.header:
            v = str(array[k])
            v_list.append(v)

        text = ",".join(v_list) + "\n"
        return text

    def log_a(self, array: dict):
        with open(self.log_path, "a") as f:
            f.write(self.dict2text(array))

    def log_w(self, array: dict):
        with open(self.log_path, "w") as f:
            f.write(self.dict2text(array))

class NpyLog:
    def __init__(self, log_dir, log_file_name, header: list):

        self.log_path = os.path.join(log_dir, log_file_name + ".npy")

        self.header = header

        self.column_num = len(self.header)

        self.np_log = []
        self.np_log.append(self.header)

    def log_a(self, array: list):
        self.np_log.append(array)
        self.save()

    def save(self):
        np.save(self.log_path, np.array(self.np_log), allow_pickle=True)

