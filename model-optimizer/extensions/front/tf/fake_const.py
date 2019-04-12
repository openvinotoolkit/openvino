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

from mo.front.common.partial_infer.utils import int64_array
from mo.front.common.replacement import FrontReplacementOp
from mo.front.tf.extractors.utils import tf_dtype_extractor
from mo.graph.graph import Node, Graph
from mo.ops.const import Const


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
