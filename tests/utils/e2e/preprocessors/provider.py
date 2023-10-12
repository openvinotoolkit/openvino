import inspect

import numpy as np
import torch

from utils.e2e.common.base_provider import BaseProvider, BaseStepProvider


class ClassProvider(BaseProvider):
    registry = {}

    @classmethod
    def validate(cls):
        methods = [
            f[0] for f in inspect.getmembers(cls, predicate=inspect.isfunction)
        ]
        if 'apply' not in methods:
            raise AttributeError(
                "Requested class {} registered as '{}' doesn't provide required method 'apply'"
                .format(cls.__name__, cls.__action_name__))


class StepProvider(BaseStepProvider):
    __step_name__ = "preprocess"

    def __init__(self, config):
        self.out_data = None
        self.executors = []
        for name, params in config.items():
            self.executors.append(ClassProvider.provide(name, params))

    def execute(self, data):
        self.out_data = data
        if isinstance(self.out_data, list):
            # case when input is torch tensor without names
            if isinstance(self.out_data[0], (torch.Tensor, np.ndarray)):
                for executor in self.executors:
                    self.out_data = executor.apply(self.out_data)
            # case of dynamism tests with --consecutive_infer key (list of two inputs)
            else:
                for executor in self.executors:
                    self.out_data = list(map(executor.apply, self.out_data))
        else:
            for executor in self.executors:
                self.out_data = executor.apply(self.out_data)
