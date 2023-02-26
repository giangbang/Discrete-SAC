import sys
import os
import torch
import datetime
import numpy as np
from collections import deque
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None  # type: ignore[misc, assignment]


class Logger:
    def __init__(self, run_name=os.path.basename(__main__.__file__).rstrip(".py"), folder="runs"):
        self.writer = SummaryWriter(f"{folder}/{run_name}")
        self.name_to_values = dict()
        self.name_to_step = dict()

    def add_hyperparams(self, hyperparams: dict):
        self.writer.add_text("hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in hyperparams.items()])),
        )

    def add_run_command(self):
        cmd = " ".join(sys.argv)
        self.writer.add_text("terminal", cmd)

    def add_scalar(self, key, val, step, smoothing=True):
        self.writer.add_scalar(key, val, step)
        if key not in self.name_to_values:
            self.name_to_values[key] = deque(maxlen=10 if smoothing else 1)
        self.name_to_values[key].extend([val])
        self.name_to_step[key] = step

    def close(self):
        self.writer.close()

    def log_stdout(self):
        results = {}
        for name, vals in self.name_to_values.items():
            results[name] = np.mean(vals)
        results["step"] = np.max(self.name_to_step.values())
        pprint(results)

    def __getitem__(self, key):
        if key not in self.name_to_values:
            self.name_to_values[key] = self._default_values()
        return self.name_to_values.get(key)

    def __setitem__(self, key, val):
        self[key].extend([val])

    def _default_values(self, deque_len=10):
        return deque(maxlen=deque_len)


def pprint(dict_data):
    '''Pretty print Hyper-parameters'''
    hyper_param_space, value_space = 40, 40
    format_str = "| {:<"+ f"{hyper_param_space}" + "} | {:<"+f"{value_space}"+"}|"
    hbar = '-'*(hyper_param_space + value_space+6)

    print(hbar)

    for k, v in dict_data.items():
        print(format_str.format(truncate_str(str(k), 40), truncate_str(str(v), 40) ) )

    print(hbar)

def truncate_str(input_str, max_length):
    """ Truncate the string if it exceeds `max_length` """
    if len(input_str) > max_length - 3:
        return input_str[:max_length-3] + "..."
    return input_str
