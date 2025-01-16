# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

import numpy as np

from openvino.tools.mo.front.common.layout import get_batch_dim, get_features_dim, shape_for_layout
from openvino.tools.mo.graph.graph import Node


def roipooling_infer(node: Node):
    """
    Sets shape of output node according specified parameters input blobs and node
    Sets number from the first input blob, channels from the second one, height and width are specified
    Parameters
    ----------
    node
    """
    shapes = [node.in_node(i).shape for i in range(len(node.in_nodes()))]
    if any(s is None for s in shapes):
        return
    if len(node.in_nodes()) == 4:  # TensorFlow case of CropAndResize operation
        crop_size = node.in_node(3).value
        if crop_size is None:
            log.error('The ROIPooling size is not known for node {}'.format(node.soft_get('name')))
            return
        if not isinstance(crop_size, np.ndarray) or len(crop_size) != 2:
            log.error('The ROIPooling size is should have 2 elements for node {}'.format(node.soft_get('name')))
        node.pooled_h = crop_size[0]
        node.pooled_w = crop_size[1]
        node.graph.remove_edge(node.in_node(3).id, node.id)
        node.graph.remove_edge(node.in_node(2).id, node.id)

    layout = node.graph.graph['layout']
    assert len(layout) == 4

    node.out_port(0).data.set_shape(shape_for_layout(layout,
                                                     batch=shapes[1][get_batch_dim(layout, 4)],
                                                     features=shapes[0][get_features_dim(layout, 4)],
                                                     height=node.pooled_h,
                                                     width=node.pooled_w))
