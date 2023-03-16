# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections import namedtuple
import numpy as np
from openvino.runtime import PartialShape, Dimension
from openvino.tools.mo.utils.error import Error
from openvino.tools.mo.utils.cli_parser import get_placeholder_shapes, split_shapes, split_inputs, get_shape_from_input_value

def get_static_shape(shape: [PartialShape, list, tuple], dynamic_value=None):
    # Current function returns list with static dimensions with following logic.
    # For dynamic dimensions return lower boundaries if they are set, otherwise
    # return upper boundaries if they are set. If dimension is fully dynamic then raise error.
    shape_list = []
    for idx, dim in enumerate(shape):
        if isinstance(dim, int):
            if dim == -1:
                shape_list.append(dynamic_value)
                continue
            shape_list.append(dim)
        elif isinstance(dim, np.int64):
            if dim == np.int64(-1):
                shape_list.append(dynamic_value)
                continue
            shape_list.append(dim)
        elif isinstance(dim, tuple):
            # tuple where (min_length, max_length), the format which uses MO cli parser
            assert len(dim) == 2, "Unknown dimension type {}".format(dim)
            if dim[0] > 0:
                shape_list.append(dim[0])
            elif dim[1] < np.iinfo(np.int64).max:
                shape_list.append(dim[1])
            else:
                shape_list.append(dynamic_value)
                continue
        elif isinstance(dim, Dimension):
            if dim.is_static or dim.get_min_length() > 0:
                shape_list.append(dim.get_min_length())
            elif dim.get_max_length() != -1:
                shape_list.append(dim.get_max_length())
            else:
                shape_list.append(dynamic_value)
                continue
        else:
            raise Error("Unknown dimension type {}".format(dim))

    return tuple(shape_list)


def get_dynamic_dims(shape: [PartialShape, list, tuple]):
    dynamic_dims = []
    for idx, dim in enumerate(shape):
        if isinstance(dim, int):
            if dim == -1:
                dynamic_dims.append(idx)
        if isinstance(dim, np.int64):
            if dim == np.int64(-1):
                dynamic_dims.append(idx)
        elif isinstance(dim, tuple):
            dynamic_dims.append(idx)
        elif isinstance(dim, Dimension):
            if dim.get_min_length() == 0 and dim.get_max_length() == -1:
                dynamic_dims.append(idx)

    return dynamic_dims


def parse_single_shape(shape, is_single_shape):
    if isinstance(shape, str):
        _, shape_tuple, _ = get_placeholder_shapes(argv_input=None, argv_input_shape=shape)
        if is_single_shape:
            raise Error("Incorrect format of shape.")
        return shape_tuple, is_single_shape
    if isinstance(shape, int) or isinstance(shape, np.int64) or isinstance(shape, Dimension):
        is_single_shape = True
    return shape, is_single_shape


def parse_input_shapes(argv):
    input_shapes = None
    if 'input_shape' in argv and argv['input_shape'] is not None:
        shapes = argv['input_shape']
        if isinstance(shapes, str):
            shapes = ["[{}]".format(x) for x in split_shapes(shapes)]
        if isinstance(shapes, (list, tuple)):
            input_shapes = []
            is_single_shape = False
            for shape in shapes:
                inp_shape, is_single_shape = parse_single_shape(shape, is_single_shape)
                input_shapes.append(inp_shape)

            if is_single_shape:
                return [input_shapes]
            else:
                return input_shapes
        elif isinstance(shapes, PartialShape):
            return [shapes]
        else:
            try:
                import torch
                if isinstance(shapes, torch.Size):
                    return [shapes]
            except ImportError:
                raise Error("Unknown type of input shape {}.".format(type(shapes)))

    if argv.get("input") is not None:
        input_shapes = parse_input_shapes_from_input(argv["input"])
    return input_shapes


def parse_input_shapes_from_input(input_info):
    if isinstance(input_info, str):
        shapes = []
        for input_value in split_inputs(input_info):
            shape = get_shape_from_input_value(input_value)
            if shape is not None:
                shapes.append(shape)
        return shapes or None
    if isinstance(input_info, tuple) and len(input_info) == 4 and isinstance(input_info[0], str):
        if input_info.shape is not None:
            return [parse_single_shape(input_info.shape, False)]
    if isinstance(input_info, (list, tuple)):
        input_shapes = []
        for inp in input_info:
            if inp.shape is not None:
                input_shapes.append(parse_single_shape(inp.shape, False))
        return input_shapes or None
    return None
