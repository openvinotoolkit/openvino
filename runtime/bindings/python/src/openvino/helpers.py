# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""openvino helper functions."""


from openvino import Function
from openvino.inference_engine import IENetwork

# TODO: will be removed after full updating MO with new api
def function_to_cnn(ng_function: Function) -> Function:
    """Get Inference Engine CNN network from openvino function."""
    capsule = Function.to_capsule(ng_function)
    return IENetwork(capsule)
