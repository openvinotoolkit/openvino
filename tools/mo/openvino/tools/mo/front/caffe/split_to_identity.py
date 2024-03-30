# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.identity import Identity
from openvino.tools.mo.front.common.replacement import FrontReplacementOp
from openvino.tools.mo.graph.graph import Graph


class SplitToIdentity(FrontReplacementOp):
    """
    The Split layer in Caffe copies input blob to a number of output layers. The Split layer in OpenVINO divides
    the input blob into several peaces. The Caffe Split layer is redundant because OpenVINO takes care of
    creation of the intermediate blobs if it is necessary.

    The replacer changes the 'op' attribute of the node to 'Identity' and set all 'out' edge attributes to be 0. So the
    Identity operations are removed further in the pipeline.
    """
    op = "Split"
    enabled = True

    def replace_sub_graph(self, graph: Graph, match: dict):
        node = match['op']

        identity = Identity(graph, {'name': node.soft_get('name', node.id)}).create_node()
        node.in_port(0).get_connection().set_destination(identity.in_port(0))

        for idx, port in node.out_ports().items():
            port.get_connection().set_source(identity.out_port(0))
