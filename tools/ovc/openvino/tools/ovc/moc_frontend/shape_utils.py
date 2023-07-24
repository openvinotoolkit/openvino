# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from openvino.runtime import PartialShape, Dimension # pylint: disable=no-name-in-module,import-error
from openvino.tools.ovc.error import Error


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
