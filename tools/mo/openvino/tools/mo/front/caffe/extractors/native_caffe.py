# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

from openvino.tools.mo.front.caffe.extractors.utils import embed_input
from openvino.tools.mo.front.common.partial_infer.caffe_fallback import caffe_native_node_infer


def blob_name(i):
    """
    Implements legacy schema for blobs naming:
        0-th blob is called 'weights'
        1-th blob is called 'biases'
    then, next blobs are called according to the new default schema
    with 'custom_' prefix: custom_2, custom_3 and so on.
    """
    predefined_names = ['weights', 'biases']
    if i < len(predefined_names):
        return predefined_names[i]
    else:
        return 'custom_{}'.format(i)


def extract_custom_blobs(node):
    """
    Enumerate all blobs in node.model_pb, for each blob
    creates a new embedded input of name 'custom_X', where X is an index >= 0 according
    to the order blobs appear in node.model_pb. The order is also enforced by input port index.
    So the order of blobs is preserved in the final IR generation.
    Order is important because they can be accessed by indices (in addition to names).
    Update node attributes in-place.
    """
    base_port = len(node.in_nodes())
    if not hasattr(node.model_pb, 'blobs'):
        return
    for i, blob in enumerate(node.model_pb.blobs):
        port = base_port + i
        internal_name = '_custom_blob_' + str(i)
        log.debug("Found new custom blob of length {} for node {}. ".format(
            len(blob.data),
            node.name if node.has_valid('name') else '<UNKNOWN>'
        ) +
                  "It will appear as input {} and internal attribute {}.".format(
                      port,
                      internal_name))
        embed_input(node.graph.node[node.id], port, internal_name, blob.data, blob_name(i))


def native_caffe_node_extractor(node):
    extract_custom_blobs(node)
    return dict(infer=caffe_native_node_infer, top=list(node.pb.top)[0])
