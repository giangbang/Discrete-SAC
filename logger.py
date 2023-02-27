import sys
import os
import torch
from datetime import datetime
import numpy as np
import time
from collections import deque
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None  # type: ignore[misc, assignment]


class Logger:
    """
    Logging class, support printing monitoring information to `std_out` and tf.event files
    """
    def __init__(self, run_name=datetime.now().strftime('%Y-%m-%d_%H%M%S'), folder="runs", tensorboard=False):
        self.tensorboard = tensorboard
        if self.tensorboard:
            self.writer = SummaryWriter(f"SAC-discrete.{folder}/{run_name}")
        self.name_to_values = dict()
        self.current_env_step = 0
        self.start_time = time.time()

    def add_hyperparams(self, hyperparams: dict):
        """
        Save hyperparameters into tensorboard
        """
        if self.tensorboard:
            self.writer.add_text("hyperparameters",
                "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in hyperparams.items()])),
            )

    def add_run_command(self):
        """
        Automatically save terminal command to tensorboard
        """
        if self.tensorboard:
            cmd = " ".join(sys.argv)
            self.writer.add_text("terminal", cmd)

    def add_scalar(self, key, val, step, smoothing=True):
        if self.tensorboard:
            self.writer.add_scalar(key, val, step)
        if key not in self.name_to_values:
            self.name_to_values[key] = deque(maxlen=10 if smoothing else 1)
        self.name_to_values[key].extend([val])
        self.current_env_step = max(self.current_env_step, step)

    def close(self):
        if self.tensorboard:
            self.writer.close()

    def log_stdout(self):
        """
        Print results to terminal
        """
        results = {}
        for name, vals in self.name_to_values.items():
            results[name] = np.mean(vals)
        results["step"] = self.current_env_step
        pprint(results)

    def __getitem__(self, key):
        if key not in self.name_to_values:
            self.name_to_values[key] = self._default_values()
        return self.name_to_values.get(key)

    def __setitem__(self, key, val):
        self[key].extend([val])

    def _default_values(self, deque_len=10):
        return deque(maxlen=deque_len)

    def fps(self):
        """
        Measuring the fps
        """
        time_pass = time.time() - self.start_time # in second
        return int(self.current_env_step / time_pass)

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
