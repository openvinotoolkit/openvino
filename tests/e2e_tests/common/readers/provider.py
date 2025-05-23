# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import inspect

from e2e_tests.common.common.base_provider import BaseProvider, BaseStepProvider


class ClassProvider(BaseProvider):
    registry = {}

    @classmethod
    def validate(cls):
        methods = [
            f[0] for f in inspect.getmembers(cls, predicate=inspect.isfunction)
        ]
        if 'read' not in methods:
            raise AttributeError(
                "Requested class {} registred as '{}' doesn't provide required method read"
                .format(cls.__name__, cls.__action_name__))


class StepProvider(BaseStepProvider):
    """
    Read network input data from the file.
    """
    __step_name__ = "read_input"

    def __init__(self, config):
        action_name = next(iter(config))
        cfg = config[action_name]
        self.executor = ClassProvider.provide(action_name, config=cfg)

    def execute(self, passthrough_data):
        model_object = passthrough_data.get('model_obj')
        passthrough_data["feed_dict"] = self.executor.read(model_object) if model_object else self.executor.read()
        passthrough_data['output'] = passthrough_data["feed_dict"]
        return passthrough_data
