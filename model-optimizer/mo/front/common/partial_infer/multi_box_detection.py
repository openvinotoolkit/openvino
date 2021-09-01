# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

from mo.front.common.partial_infer.utils import is_fully_defined, compatible_dims
from mo.graph.graph import Node
from mo.utils.error import Error


def multi_box_detection_infer(node: Node):
    loc_shape = node.in_node(0).shape
    conf_shape = node.in_node(1).shape
    prior_boxes_shape = node.in_node(2).shape
    node_name = node.soft_get('name', node.id)

    if loc_shape is None or conf_shape is None or prior_boxes_shape is None:
        raise Error('Shapes for the Detection Output node "{}" are not defined'.format(node_name))

    prior_size = 4
    if node.has('normalized') and not node.normalized:
        prior_size = 5

    if is_fully_defined(prior_boxes_shape[-1]) and prior_boxes_shape[-1] % prior_size != 0:
        raise Error('Amount of confidences "{}" is not divisible by {} for node "{}"'
                    ''.format(prior_boxes_shape[-1], prior_size, node_name))

    num_priors = prior_boxes_shape[-1] // prior_size
    if not node.has_valid('keep_top_k') or node.keep_top_k == -1:
        node['keep_top_k'] = num_priors

    # do not try to infer number of classes because it is not possible in case when input shapes are partially defined
    if not node.has_valid('num_classes'):
        node['num_classes'] = conf_shape[-1] // num_priors
        log.debug('Inferred amount of classes "{}"'.format(node.num_classes))

    num_loc_classes = node.num_classes
    if node.has_and_set('share_location') and node.share_location:
        num_loc_classes = 1

    if not compatible_dims(num_priors * num_loc_classes * 4, loc_shape[-1]):
        raise Error('Locations and prior boxes shapes mismatch: "{}" vs "{}" for node "{}"'
                    ''.format(loc_shape, prior_boxes_shape, node_name))

    if not node.variance_encoded_in_target and not compatible_dims(prior_boxes_shape[-2], 2):
        raise Error('The "-2" dimension of the prior boxes must be 2 but it is "{}" for node "{}".'
                    ''.format(prior_boxes_shape[-2], node_name))

    if is_fully_defined(conf_shape[-1]) and is_fully_defined(num_priors) and conf_shape[-1] % num_priors != 0:
        raise Error('Amount of confidences "{}" is not divisible by amount of priors "{}" for node "{}".'
                    ''.format(conf_shape[-1], num_priors, node_name))

    node.out_port(0).data.set_shape([1, 1, conf_shape[0] * node.keep_top_k, 7])

    # the line below is needed for the TF framework so the MO will not change the layout
    node.graph.node[node.out_node(0).id]['nchw_layout'] = True
