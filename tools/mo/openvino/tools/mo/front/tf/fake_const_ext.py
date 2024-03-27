# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.common.replacement import FrontReplacementOp
from openvino.tools.mo.front.tf.extractors.utils import tf_dtype_extractor
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.ops.const import Const


class FakeConstToConst(FrontReplacementOp):
    op = "FakeConst"
    enabled = True

    def replace_sub_graph(self, graph: Graph, match: dict):
        node = match['op']
        if not node.has_valid('value'):
            log.debug("No value in FakeConst node {}".format(node.id))
            return
        node_value = node.value
        extracted_attrs = {
            'data_type': tf_dtype_extractor(node.pb.attr['dtype'].type),
            'shape': int64_array(node_value.shape),
            'value': node_value
        }
        Const.update_node_stat(node, extracted_attrs)
        log.debug('FakeConst op was translated to Const op with shape = {} and value.shape = {}'
                  ''.format(extracted_attrs['shape'], extracted_attrs['value'].shape))
