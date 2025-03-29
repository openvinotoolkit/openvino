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
        if 'get_ir' not in methods:
            raise AttributeError(
                "Requested class {} registered as '{}' doesn't provide required method get_ir"
                .format(cls.__name__, cls.__action_name__))


class StepProvider(BaseStepProvider):
    __step_name__ = "get_ir"

    def __init__(self, config):
        action_name = next(iter(config))
        cfg = config[action_name]
        self.executor = ClassProvider.provide(action_name, config=cfg)

    def execute(self, passthrough_data):
        # this may be considered a WA. To properly remove prepared_model
        # we need to refactor all the class providers and handle pytorch cases with care
        self.executor.prepared_model = passthrough_data.get("model_obj")
        data = passthrough_data.get('feed_dict')
        passthrough_data['xml'], passthrough_data['bin'] = self.executor.get_ir(data)
        # passthrough_data['mo_log'] = self.executor.mo_log
        return passthrough_data
