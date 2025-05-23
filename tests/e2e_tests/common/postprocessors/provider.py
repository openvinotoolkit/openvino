# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import inspect

import torch

from e2e_tests.common.common.base_provider import BaseProvider, BaseStepProvider


class ClassProvider(BaseProvider):
    registry = {}

    @classmethod
    def validate(cls):
        methods = [
            f[0] for f in inspect.getmembers(cls, predicate=inspect.isfunction)
        ]
        if 'apply' not in methods:
            raise AttributeError(
                "Requested class {} registred as '{}' doesn't provide required method 'apply'"
                .format(cls.__name__, cls.__action_name__))


class StepProvider(BaseStepProvider):
    __step_name__ = "postprocessor"

    def __init__(self, config):
        self.executors = []
        for name, params in config.items():
            self.executors.append(ClassProvider.provide(name, params))

    def execute(self, passthrough_data):
        data = passthrough_data.strict_get('output', self)
        if isinstance(data, list):
            # case when input is torch tensor without names
            if isinstance(data[0], torch.Tensor):
                for executor in self.executors:
                    data = executor.apply(data)
            # case of dynamism tests with --consecutive_infer key (list of two inputs)
            else:
                for executor in self.executors:
                    data = list(map(executor.apply, data))
        else:
            for executor in self.executors:
                data = executor.apply(data)
        passthrough_data['output'] = data
        return passthrough_data
