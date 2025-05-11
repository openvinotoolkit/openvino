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
        if 'load_model' not in methods:
            raise AttributeError(
                "Requested class {} registered as '{}' doesn't provide required method load_model"
                .format(cls.__name__, cls.__action_name__))


class StepProvider(BaseStepProvider):
    __step_name__ = "load_model"

    def __init__(self, config):
        action_name = next(iter(config))
        cfg = config[action_name]
        self.executor = ClassProvider.provide(action_name, config=cfg)

    def execute(self, passthrough_data):
        data = passthrough_data.get('feed_dict')
        passthrough_data['model_obj'] = self.executor.load_model(data)
        passthrough_data['output'] = passthrough_data['model_obj']
        return passthrough_data
