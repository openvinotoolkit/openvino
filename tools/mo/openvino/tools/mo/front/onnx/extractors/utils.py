# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.graph.graph import Node
from openvino.tools.mo.utils.error import Error


def onnx_node_has_attr(node: Node, name: str):
    attrs = [a for a in node.pb.attribute if a.name == name]
    return len(attrs) != 0


def onnx_attr(node: Node, name: str, field: str, default=None, dst_type=None):
    """ Retrieves ONNX attribute with name `name` from ONNX protobuf `node.pb`.
        The final value is casted to dst_type if attribute really exists.
        The function returns `default` otherwise.
    """
    attrs = [a for a in node.pb.attribute if a.name == name]
    if len(attrs) == 0:
        # there is no requested attribute in the protobuf message
        return default
    elif len(attrs) > 1:
        raise Error('Found multiple entries for attribute name {} when at most one is expected. Protobuf message with '
                    'the issue: {}.', name, node.pb)
    else:
        res = getattr(attrs[0], field)
        if dst_type is not None:
            return dst_type(res)
        else:
            return res


def onnx_get_num_outputs(node: Node):
    """ Retrieves number of outputs for ONNX operation.
    """
    return len(node.pb.output)


def get_backend_pad(pads, spatial_dims, axis):
    return [x[axis] for x in pads[spatial_dims]]


def get_onnx_autopad(auto_pad):
    auto_pad = auto_pad.decode().lower()
    if auto_pad == 'notset':
        auto_pad = None
    return auto_pad


def get_onnx_opset_version(node: Node):
    return node.graph.graph.get('fw_opset_version', 0)


def get_onnx_datatype_as_numpy(value):
    datatype_to_numpy = {
        1: np.float32,
        9: bool,
        11: np.double,
        10: np.float16,
        5: np.int16,
        6: np.int32,
        7: np.int64,
        3: np.int8,
        8: np.ubyte,
        4: np.uint16,
        12: np.uint32,
        13: np.uint64,
        2: np.uint8,
    }
    try:
        return datatype_to_numpy[value]
    except KeyError:
        raise Error("Incorrect value {} for Datatype enum".format(value))
