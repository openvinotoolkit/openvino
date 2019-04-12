"""
 Copyright (c) 2018-2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import numpy as np

from mo.graph.graph import Node
from mo.utils.error import Error


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


def get_backend_pad(pads, spatial_dims, axis):
    return [x[axis] for x in pads[spatial_dims]]


def get_onnx_autopad(auto_pad):
    auto_pad = auto_pad.decode().lower()
    if auto_pad == 'notset':
        auto_pad = None
    return auto_pad


def get_onnx_datatype_as_numpy(value):
    datatype_to_numpy = {
                         1: np.float32,
                         9: np.bool,
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
