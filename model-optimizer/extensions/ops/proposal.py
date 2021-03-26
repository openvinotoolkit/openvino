# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.front.common.partial_infer.utils import int64_array
from mo.front.extractor import attr_getter, bool_to_str
from mo.graph.graph import Node, Graph
from mo.ops.op import Op


class ProposalOp(Op):
    op = 'Proposal'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': __class__.op,
            'op': __class__.op,
            'version': 'opset4',
            'post_nms_topn': 300,  # default in caffe-shared
            'infer': ProposalOp.proposal_infer,
            'in_ports_count': 3,
            'out_ports_count': 1 if attrs.get('version') == 'opset1' else 2,
            'normalize': False,
            'clip_before_nms': True,
            'clip_after_nms': False,
        }
        super().__init__(graph, mandatory_props, attrs)

    def supported_attrs(self):
        return [
            'feat_stride',
            'base_size',
            'min_size',
            'ratio',
            'scale',
            'pre_nms_topn',
            'post_nms_topn',
            'nms_thresh',
        ]

    def backend_attrs(self):
        return [
            'feat_stride',
            'base_size',
            'min_size',
            ('ratio', lambda node: attr_getter(node, 'ratio')),
            ('scale', lambda node: attr_getter(node, 'scale')),
            'pre_nms_topn',
            'post_nms_topn',
            'nms_thresh',
            'framework',
            'box_coordinate_scale',
            'box_size_scale',
            ('normalize', lambda node: bool_to_str(node, 'normalize')),
            ('clip_after_nms', lambda node: bool_to_str(node, 'clip_after_nms')),
            ('clip_before_nms', lambda node: bool_to_str(node, 'clip_before_nms')),
        ]

    @staticmethod
    def proposal_infer(node: Node):
        input_shape = node.in_node(0).shape
        out_shape = int64_array([input_shape[0] * node.post_nms_topn, 5])
        # rois blob: holds R regions of interest, each is a 5 - tuple
        # (n, x1, y1, x2, y2) specifying an image batch index n and a
        # rectangle(x1, y1, x2, y2)
        node.out_port(0).data.set_shape(out_shape)

        # the second optional output contains box probabilities
        if len(node.out_ports()) == 2 and not node.out_port(1).disconnected():
            node.out_port(1).data.set_shape(int64_array([input_shape[0] * node.post_nms_topn]))
