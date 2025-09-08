# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import inspect
from e2e_tests.common.common.base_provider import BaseProvider


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


class Container:
    def __init__(self, config):
        self.executors = []
        for name, params in config.items():
            self.executors.append(ClassProvider.provide(name, params))

    def execute(self, network, **kwargs):
        for executor in self.executors:
            executor.apply(network, **kwargs)
