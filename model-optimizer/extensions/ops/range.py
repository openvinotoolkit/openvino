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

from mo.graph.graph import Node, Graph
from mo.ops.op import Op


class Range(Op):
    op = 'Range'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': __class__.op,
            'op': __class__.op,
            'in_ports_count': 3,
            'out_ports_count': 1,
            'infer': __class__.infer,
        }
        super().__init__(graph, mandatory_props, attrs)

    @staticmethod
    def infer(node: Node):
        start = node.in_node(0)
        limit = node.in_node(1)
        delta = node.in_node(2)
        output = node.out_node()

        if not start.has_valid('value') or not limit.has_valid('value') or not delta.has_valid('value'):
            log.error("Range operation is supported with constant inputs only")
            return
        if node.has_valid('pb') and 'type' in node.pb.attr:
            from mo.front.tf.extractors.utils import tf_dtype_extractor
            result_data_type = tf_dtype_extractor(node.pb.attr["type"].type)
        else:
            result_data_type = start.value.dtype
        output.value = np.arange(start.value, limit.value, delta.value, dtype=result_data_type)
        output.shape = np.array(output.value.shape, dtype=np.int64)

        # Some notes on the automatic result data type infer. The tf.range does is differently than np.arange. Numpy
        # by default creates array with elements of type int64 and float64, but TF does not widen data types and keep them
        # int32 and float32.
        # Compare:

        # >>> tf.range(1, 5, 0.5)
        # <tf.Tensor 'range_1:0' shape = (8,) dtype = float32>
        # >>> tf.range(1, 5, 2)
        # <tf.Tensor 'range_2:0' shape = (2,) dtype = int32>

        # >>> np.array([0.5], dtype=np.float32)
        # array([0.5], dtype=float32)
        # >>> np.arange(np.array([1], dtype=np.int32), np.array([5], dtype=np.int32), np.array([2], dtype=np.int32)).dtype
        # dtype('int64')
        # >>> np.arange(np.array([1], dtype=np.int32), np.array([5], dtype=np.int32), np.array([0.5], dtype=np.float32)).dtype
        # dtype('float64')
