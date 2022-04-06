# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, List, Union

import numpy as np

import ngraph as ng
from ngraph.utils.types import NumericData
from tests_compatibility.runtime import get_runtime
from string import ascii_uppercase


def _get_numpy_dtype(scalar):
    return np.array([scalar]).dtype


def run_op_node(input_data, op_fun, *args):
    # type: (Union[NumericData, List[NumericData]], Callable, *Any) -> List[NumericData]
    """Run computation on node performing `op_fun`.

    `op_fun` has to accept a node as an argument.

    This function converts passed raw input data to nGraph Constant Node and that form is passed
    to `op_fun`.

    :param input_data: The input data for performed computation.
    :param op_fun: The function handler for operation we want to carry out.
    :param args: The arguments passed to operation we want to carry out.
    :return: The result from computations.
    """
    runtime = get_runtime()
    comp_args = []
    op_fun_args = []
    comp_inputs = []

    for idx, data in enumerate(input_data):
        node = None
        if np.isscalar(data):
            node = ng.parameter([], name=ascii_uppercase[idx], dtype=_get_numpy_dtype(data))
        else:
            node = ng.parameter(data.shape, name=ascii_uppercase[idx], dtype=data.dtype)
        op_fun_args.append(node)
        comp_args.append(node)
        comp_inputs.append(data)

    op_fun_args.extend(args)
    node = op_fun(*op_fun_args)
    computation = runtime.computation(node, *comp_args)
    return computation(*comp_inputs)


def run_op_numeric_data(input_data, op_fun, *args):
    # type: (NumericData, Callable, *Any) -> List[NumericData]
    """Run computation on node performing `op_fun`.

    `op_fun` has to accept a scalar or an array.

    This function passess input data AS IS. This mean that in case they're a scalar (integral,
    or floating point value) or a NumPy's ndarray object they will be automatically converted
    to nGraph's Constant Nodes.

    :param input_data: The input data for performed computation.
    :param op_fun: The function handler for operation we want to carry out.
    :param args: The arguments passed to operation we want to carry out.
    :return: The result from computations.
    """
    runtime = get_runtime()
    node = op_fun(input_data, *args)
    computation = runtime.computation(node)
    return computation()


def count_ops_of_type(func, op_type):
    count = 0
    for op in func.get_ops():
        if (type(op) is type(op_type)):
            count += 1
    return count
