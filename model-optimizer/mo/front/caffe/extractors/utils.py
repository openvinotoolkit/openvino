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


def dim_to_shape(dim):
    """
    Extracts proto message with shape dimensions to shape expressed as np.array.
    Args:
        dim: proto message with shape dimensions

    Returns:
        shape of the layer as np.array
    """
    return np.array(dim, dtype=np.int64)


def embed_input(attrs: dict, port: int, name: str, value: np.array, bin_name: str = None):
    """
    Appends port information to the given set of attributes of the current layer.
    Mutates passed attributes.
    Args:
        attrs: dictionary of existing attributes
        port: relative number of the port for the layer
        name: name of the input
        value: np.array of values
        bin_name: optional, representing the specific behavior of the blob,
         either 'weights' or 'biases'

    Returns:
        mutated attributes dictionary with new properties under 'embedded_inputs' key

    """
    assert name not in attrs
    attrs[name] = np.array(value)

    if 'embedded_inputs' not in attrs:
        attrs['embedded_inputs'] = []
    if not bin_name:
        bin_name = name
    input_val = (port, name, {'bin': bin_name})
    # (input index, input name, future edge attributes)
    attrs['embedded_inputs'].append(input_val)  # pylint: disable=not-callable


def weights_biases(bias_term: bool, model_layer, start_index: int = 1, proto={}):
    """
    Creates object with configured inputs in the following order: 0: weights, 1: biases
    Args:
        bias_term: flag to whether include biases in the final input or not
        model_layer: caffemodel layer containing values in blobs

    Returns:
        dictionary with set up inputs or empty dictionary
    """
    attrs = {}
    if not model_layer:
        if proto != {}:
            if proto.weight_filler:
                if proto.weight_filler.type == "diagonal":
                    data_len = proto.kernel_size[0] * proto.kernel_size[0] * proto.num_output
                    data = np.zeros(data_len * data_len, dtype=np.float32)
                    for i in range(0, data_len):
                        data[i * (data_len + 1)] = proto.weight_filler.diag_val[i]

                    bias = np.zeros(proto.num_output, np.float32)
                    embed_input(attrs, start_index, 'weights', data)
                    if bias_term:
                        embed_input(attrs, start_index + 1, 'biases', bias)

        return attrs

    blobs = model_layer.blobs
    embed_input(attrs, start_index, 'weights', blobs[0].data)
    if bias_term:
        embed_input(attrs, start_index + 1, 'biases', blobs[1].data)
    return attrs


def get_list_from_container(param, prop: str, t):
    """
    Takes proto parameter and extracts a value it stores.
    Args:
        param: proto parameter
        prop: name of the property to take
        t: type of the value (int, float etc.) - only primitive ones

    Returns:
        If it is a container, returns the list with values.
        If it is a single value of the given type - a list of single value.
        If neither or property does not exist for param - empty list.
    """
    if not param or (param and not hasattr(param, prop)):
        return []

    prop_val = getattr(param, prop)

    if not prop_val:
        return []
    elif isinstance(prop_val, t):
        return [prop_val]
    elif len(prop_val) > 0:
        return prop_val
    return []


def get_spatial_attr(default: list, single_name: str, name: str, param):
    attr_h = default[1]
    attr_w = default[0]
    if hasattr(param, '{}_h'.format(name)):
        if getattr(param, '{}_h'.format(name)) != default[1] and getattr(param, '{}_h'.format(name)) != 0:
            attr_h = getattr(param, '{}_h'.format(name))
    if hasattr(param, '{}_w'.format(name)):
        if getattr(param, '{}_w'.format(name)) != default[0] and getattr(param, '{}_w'.format(name)) != 0:
            attr_w = getattr(param, '{}_w'.format(name))
    if (not attr_h or not attr_w) or (attr_h == attr_w == default[0]):
        attrs = get_list_from_container(param, single_name, int)
        if len(attrs) > 0 and attrs != default:
            attr_w = attr_h = attrs[0]
    return attr_w, attr_h


def get_canonical_axis_index(shape, axis):
    return len(shape) + axis if axis < 0 else axis
