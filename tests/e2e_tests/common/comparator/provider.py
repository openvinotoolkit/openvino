# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import inspect
from e2e_tests.common.common.base_provider import BaseProvider


class ClassProvider(BaseProvider):
    __step_name__ = "compare"
    registry = {}

    @classmethod
    def validate(cls):
        methods = [
            f[0] for f in inspect.getmembers(cls, predicate=inspect.isfunction)
        ]
        if 'compare' not in methods:
            raise AttributeError(
                "Requested class {} registred as '{}' doesn't provide required method compare"
                .format(cls.__name__, cls.__action_name__))
