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
        if 'get_refs' not in methods:
            raise AttributeError(
                "Requested class {} registred as '{}' doesn't provide required method get_refs"
                .format(cls.__name__, cls.__action_name__))


class TFHubStepProvider(BaseStepProvider):
    __step_name__ = "get_refs_tf_hub"

    def __init__(self, config):
        action_name = next(iter(config))
        cfg = config[action_name]
        self.executor = ClassProvider.provide(action_name, config=cfg)

    def execute(self, passthrough_data):
        passthrough_data['output'] = self.executor.get_refs(passthrough_data)
        return passthrough_data
