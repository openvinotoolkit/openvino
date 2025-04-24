# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import inspect
from e2e_tests.test_utils.test_utils import log_timestamp
from e2e_tests.common.common.base_provider import BaseProvider, BaseStepProvider


class ClassProvider(BaseProvider):
    registry = {}

    @classmethod
    def validate(cls):
        methods = [
            f[0] for f in inspect.getmembers(cls, predicate=inspect.isfunction)
        ]
        if 'infer' not in methods:
            raise AttributeError(
                "Requested class {} registred as '{}' doesn't provide required method infer"
                .format(cls.__name__, cls.__action_name__))


class StepProvider(BaseStepProvider):
    __step_name__ = "infer"

    def __init__(self, config):
        action_name = next(iter(config))
        self.executor = ClassProvider.provide(action_name, config=config[action_name])

    def execute(self, passthrough_data=None):
        feed_dict = passthrough_data.strict_get('feed_dict', self)
        self.executor.xml, self.executor.bin = passthrough_data.get('xml'), passthrough_data.get('bin')
        with log_timestamp('Inference'):
            passthrough_data['output'] = self.executor.infer(feed_dict)
        return passthrough_data
