# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# This is a wrapper to mark a function that is used as jax module.
# It's a temporary workaround to avoid mistaking a normal function as jax module.
# see `tools/ovc/openvino/tools/ovc/convert_impl.py:check_model_object`
def jax_debug(func):
    func.__jax_debug__ = True
    return func
