# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import dynamic_dimension_value
from openvino.tools.mo.front.common.partial_infer.utils import mo_array, int64_array
from openvino.tools.mo.graph.graph import Node
from openvino.tools.mo.utils.error import Error

nchw_to_nhwc_permute = int64_array([0, 2, 3, 1])
nhwc_to_nchw_permute = int64_array([0, 3, 1, 2])
supported_layouts = ('NCHW', 'NHWC')
# the attribute 'layout' in the graph.graph can have two values only: "NCHW" or "NHWC". If the tensor has 5 dimensions
# then it is necessary to transform "NCHW" to "NCDHW" and "NHWC" to "NDHWC" respectively. The dictionary below id used
# for this purpose.
indices_mapping = {4: {'NCHW': 'NCHW',
                       'NHWC': 'NHWC'},
                   5: {'NCHW': 'NCDHW',
                       'NHWC': 'NDHWC'}}


def convert_shape(shape: np.array, permute: np.array):
    result = [0, 0, 0, 0]
    for ind, perm_ind in enumerate(permute):
        result[ind] = shape[perm_ind]
    return mo_array(result)


def get_depth_dim(layout: str, shape_len: int):
    """
    Gets index of the dimension corresponding to depth.
    :param layout: string representing layout: NCHW or NHWC usually.
    :param shape_len: the shape length.
    :return: index of the 'D' character
    """
    assert layout in supported_layouts
    assert shape_len == 5
    return indices_mapping[shape_len][layout].find('D')


def get_height_dim(layout: str, shape_len: int):
    """
    Gets index of the dimension corresponding to height.
    :param layout: string representing layout: NCHW or NHWC usually.
    :param shape_len: the shape length.
    :return: index of the 'H' character
    """
    assert layout in supported_layouts
    assert 4 <= shape_len <= 5
    return indices_mapping[shape_len][layout].find('H')


def get_width_dim(layout: str, shape_len: int):
    """
    Gets index of the dimension corresponding to width.
    :param layout: string representing layout: NCHW or NHWC usually.
    :param shape_len: the shape length.
    :return: index of the 'W' character
    """
    assert layout in supported_layouts
    assert 4 <= shape_len <= 5
    return indices_mapping[shape_len][layout].find('W')


def get_features_dim(layout: str, shape_len: int):
    """
    Gets index of the dimension corresponding to features.
    :param layout: string representing layout: NCHW or NHWC usually.
    :param shape_len: the shape length.
    :return: index of the 'C' character
    """
    assert layout in supported_layouts
    assert 4 <= shape_len <= 5
    return indices_mapping[shape_len][layout].find('C')


def get_batch_dim(layout: str, shape_len: int):
    """
    Gets index of the dimension corresponding to batch.
    :param layout: string representing layout: NCHW or NHWC usually.
    :param shape_len: the shape length.
    :return: index of the 'N' character
    """
    assert layout in supported_layouts
    assert 4 <= shape_len <= 5
    return indices_mapping[shape_len][layout].find('N')


def shape_for_layout(layout: str, **kwargs):
    """
    Creates 4D or 5D tensor with the layout with specified dimension sizes.
    :param layout: layout string.
    :param kwargs: dictionary that contains the dimension sizes using the following keys: 'batch', 'features', 'depth',
    'height', 'width'.
    :return: shape_array of type np.int64 with 4 or 5 elements.
    """
    assert layout in supported_layouts
    for required_key in ('batch', 'features', 'height', 'width'):
        if required_key not in kwargs:
            raise Error('Required parameter "{}" is missing.'.format(required_key))
    for key in kwargs.keys():
        if key not in ('batch', 'features', 'height', 'width', 'depth'):
            raise Error('Parameter "{}" is not supported.'.format(key))

    depth = kwargs.get('depth', None)
    shape_len = 4 + (depth is not None)
    output_shape = np.ma.ones(shape=[shape_len], dtype=np.int64, fill_value=dynamic_dimension_value)
    output_shape[get_batch_dim(layout, shape_len)] = kwargs['batch']
    output_shape[get_height_dim(layout, shape_len)] = kwargs['height']
    output_shape[get_width_dim(layout, shape_len)] = kwargs['width']
    output_shape[get_features_dim(layout, shape_len)] = kwargs['features']
    if depth is not None:
        output_shape[get_depth_dim(layout, shape_len)] = depth
    return output_shape


def get_dim_from_layout(node: Node, dim: str):
    """
    Gets index of dimension from layout specified for node.
    :param node: node to get dim for.
    :param dim: name of dimension to get index for.
    :return: tuple with index of the dimension and bool flag if the node has layout specified or no.
    """
    layout = None
    graph = node.graph
    if 'layout_values' in graph.graph['cmd_params'] and graph.graph['cmd_params'].layout_values:
        layout_values = graph.graph['cmd_params'].layout_values.copy()
        if '' in layout_values:
            in_nodes = graph.get_op_nodes(op='Parameter')
            if len(in_nodes) == 1:
                in_node = in_nodes[0]
                layout_values[in_node.soft_get('name', in_node.id)] = layout_values['']
                del layout_values['']
        name = node.soft_get('name', node.id)
        if name in layout_values:
            if layout_values[name]['source_layout']:
                layout = layout_values[name]['source_layout']

    if layout:
        from openvino.runtime import Layout  # pylint: disable=no-name-in-module,import-error

        layout_parsed = Layout(layout)
        has_dim = layout_parsed.has_name(dim)
        if has_dim:
            idx = layout_parsed.get_index_by_name(dim)
            if idx < 0:
                idx = len(node.shape) + idx
            return idx, True
        else:
            return None, True
    else:
        return None, False
