# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Dummy infer provider to be used when real provider is unavailable due to absence of IE Python API
e.g. for IR collection environment
"""
from .provider import ClassProvider


def use_dummy(name, message):
    class DummyInfer(ClassProvider):
        __action_name__ = name

        def __init__(self, *args, **kwargs):
            pass

        def infer(self, *args, **kwargs):
            raise RuntimeError(message)

    return DummyInfer
