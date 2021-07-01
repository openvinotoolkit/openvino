# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

import numpy as np

from mo.front.extractor import attr_getter
from mo.graph.graph import Node, Graph
from mo.ops.op import Op


class SimplerNMSOp(Op):
    op = 'SimplerNMS'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': __class__.op,
            'op': __class__.op,
            'version': 'experimental',
            'in_ports_count': 3,
            'out_ports_count': 1,
            'infer': SimplerNMSOp.simplernms_infer
        }
        super().__init__(graph, mandatory_props, attrs)

    def supported_attrs(self):
        return [
            'cls_threshold',
            'max_num_proposals',
            'iou_threshold',
            'min_bbox_size',
            'feat_stride',
            'pre_nms_topn',
            'post_nms_topn',
            'scale'
        ]

    def backend_attrs(self):
        return [
            'cls_threshold',
            'max_num_proposals',
            'iou_threshold',
            'min_bbox_size',
            'feat_stride',
            'pre_nms_topn',
            'post_nms_topn',
            ('scale', lambda node: attr_getter(node, 'scale'))
        ]

    @staticmethod
    def simplernms_infer(node: Node):
        """
           Sets shape of output node according to specified param of post_nms_topn
           and number of the following params: [is_obj, x, y, w, h]
           Parameters
           ----------
           node

           """
        if node.feat_stride != 16:
            log.error("SimplerNMS layer doesn't support other feat_stride value that 16")
            return

        scale_list = []
        for i in range(0, len(node.scale)):
            scale_list.append(str(node.scale[i]))

        node.scale = scale_list

        node.out_node().shape = np.array([node.post_nms_topn, 5])
