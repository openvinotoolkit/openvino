"""
 Copyright (C) 2018-2020 Intel Corporation

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

from extensions.back.ForceStrictPrecision import ForceStrictPrecision
from extensions.back.ReduceToPooling import ReduceReplacer
from mo.back.replacement import BackReplacementPattern
from mo.front.common.partial_infer.utils import int64_array
from mo.front.tf.graph_utils import create_op_node_with_second_input
from mo.graph.graph import Graph
from mo.ops.squeeze import Squeeze


class Reshape0DToSqueeze(BackReplacementPattern):
    """
    Transformation looks for the Reshape layers which generate 0D tensor and replace them with Reshape_1D->Squeeze to
    overcome issue the IE doesn't 1D constants with value [0] which is generated for the Reshape to OD case.
    """
    enabled = True
    run_not_recursively = True
    force_shape_inference = True

    def run_after(self):
        return [ReduceReplacer]

    def run_before(self):
        return [ForceStrictPrecision]

    def find_and_replace_pattern(self, graph: Graph):
        for reshape_node in graph.get_op_nodes(op='Reshape'):
            output_shape = reshape_node.in_port(1).data.get_value()
            shape_producer_node = reshape_node.in_port(1).get_source().node
            assert output_shape is not None
            if np.array_equal(output_shape, []) and shape_producer_node.op == 'Const':
                log.debug('Reshape node {} changes shape to 0D tensor.'.format(reshape_node.name))
                shape_producer_node.value = int64_array([1])
                shape_producer_node.shape = int64_array([1])
                shape_producer_node['need_shape_inference'] = True
                shape_producer_node['override_output_shape'] = True

                reshape_node['need_shape_inference'] = True
                reshape_node['override_output_shape'] = True

                squeeze_0D = create_op_node_with_second_input(graph, Squeeze, int64_array([0]))
                squeeze_0D['override_output_shape'] = True

                reshape_node.out_port(0).get_connection().insert_node(squeeze_0D)
