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
import logging as log

import numpy as np

from mo.front.caffe.extractors.utils import get_canonical_axis_index
from mo.front.common.layout import get_height_dim, get_width_dim, get_depth_dim
from mo.front.common.partial_infer.utils import int64_array
from mo.ops.op import PermuteAttrs
from mo.utils.error import Error


def is_spatial_squeeze(layout: str, input_shape: np.ndarray, squeeze_dims: np.ndarray):
    """
    Checks that the squeeze operation removes all spatial dimensions.
    :param layout: graph layout.
    :param input_shape: numpy array with input shape.
    :param squeeze_dims: numpy array with dims to squeeze.
    :return: result of the check.
    """
    if len(input_shape) < 4 or len(input_shape) > 5:
        return False
    spatial_dims = [get_height_dim(layout, len(input_shape)), get_width_dim(layout, len(input_shape))]
    if len(input_shape) == 5:
        spatial_dims.append(get_depth_dim(layout, len(input_shape)))
    for dim in spatial_dims:
        if input_shape[dim] != 1:
            log.debug('The reshape from "{}" with squeezed dims "{}" is not a spatial squeeze'.format(input_shape,
                                                                                                      squeeze_dims))
            return False
    if len(squeeze_dims) != len(spatial_dims):
        log.debug('The reshape from "{}" with squeezed dims "{}" is not a spatial squeeze'.format(input_shape,
                                                                                                  squeeze_dims))
        return False
    log.debug('The reshape from "{}" with squeezed dims "{}" is not a spatial squeeze'.format(input_shape,
                                                                                              squeeze_dims))
    return True


def tf_squeeze_infer(node):
    if node.squeeze_dims is None:
        # TODO: implement; there is no implementation now because no test
        return

    real_squeeze_dims = []
    input_shape = node.in_node().shape
    if input_shape is None:
        return
    # UGLY
    output_shape = input_shape.copy()
    for n in node.squeeze_dims:
        if output_shape[n] == 1:
            real_squeeze_dims.append(get_canonical_axis_index(output_shape, n))
        else:
            raise Error('Trying to squeeze dimension not equal to 1 for node "{}"'.format(node.soft_get('name')))

    output_shape = np.delete(output_shape, real_squeeze_dims)
    node.out_node().shape = output_shape

    if is_spatial_squeeze(node.graph.graph['layout'], input_shape, output_shape):
        output_shape = int64_array([0, -1])
    node['dim'] = output_shape
    if node.in_node().value is not None:
        node.out_node().value = np.array(np.reshape(node.in_node().value, output_shape))

    PermuteAttrs.create_permute_attrs(node, attrs=[('dim', 'output:0')])
