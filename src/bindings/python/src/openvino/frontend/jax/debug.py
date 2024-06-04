# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa
# mypy: ignore-errors

def jax_debug(func):
    '''
    This is a wrapper to mark a function that is used as jax module.
    It's a temporary workaround to avoid mistaking a normal function as jax module.
    see `tools/ovc/openvino/tools/ovc/convert_impl.py:check_model_object`
    '''
    func.__jax_debug__ = True
    return func
