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


def swap_weights_xy(nodes: list):
    """
    The function changes weights of the nodes from the 'nodes' list which are used with calculations with coordinates of
    some objects. The function should be used when it is necessary to virtually change the layout of data from XY to YX.
    The node from the 'nodes' list should be some sort of convolution node or matrix multiplication.
    The function also swaps weights in the following Add and BiasAdd operations.
    :param nodes: list of Node objects to change the weights in them.
    :return: None
    """
    for node in nodes:
        weights_node = node.in_node(1)
        if weights_node.has_and_set('swap_xy_count'):
            weights_node['swap_xy_count'] += 1
            log.debug('Increasing value for attribute "swap_xy_count" to {} for node {}'.format(
                weights_node['swap_xy_count'], weights_node.name))
            continue
        weights_node['swap_xy_count'] = 1
        log.debug('Swapping weights for node "{}"'.format(node.name))
        reshaped_weights = weights_node.value.reshape((-1, 2))
        new_swapped_weights = np.concatenate((reshaped_weights[:, 1:2], reshaped_weights[:, 0:1]), 1)
        new_swapped_weights = new_swapped_weights.reshape(weights_node.shape)
        weights_node.shape = np.array(new_swapped_weights.shape, dtype=np.int64)
        weights_node.value = new_swapped_weights
        # biases
        for m in node.out_node().out_nodes():
            if m.has_valid('op') and m.op in ['Add']:
                biases = m.in_node(1).value  # biases are weights of the (Bias)Add op
                # swap Y and X to the regular order that the IE expects
                reshaped_biases = biases.reshape((-1, 2))
                swapped_biases = np.concatenate((reshaped_biases[:, 1:2], reshaped_biases[:, 0:1]), 1)
                # reshaping back to the original shape
                swapped_biases = swapped_biases.reshape(biases.shape)
                m.in_node(1).shape = np.array(swapped_biases.shape, dtype=np.int64)
                m.in_node(1).value = swapped_biases
