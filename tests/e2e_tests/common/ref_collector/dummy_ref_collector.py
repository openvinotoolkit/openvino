# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Dummy reference collector to be used when real collector is unavailable.

Example usage: User do not have tensorflow installed and want to run pytorch-only
tests. Tensorflow reference collector is substituted by the dummy so that no
error occurs during pytest test collection/run. If user try to run
tensorflow-related tests, the execution fails due to error specified.
"""
from e2e_tests.common.ref_collector.provider import ClassProvider


def use_dummy(name, error_message):
    class DummyRefCollector(ClassProvider):
        __action_name__ = name

        def __init__(self, *args, **kwargs):
            raise ValueError(error_message)

        def get_refs(self, *args, **kwargs):
            pass

    return DummyRefCollector
