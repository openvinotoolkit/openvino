"""
 Copyright (C) 2020 Intel Corporation

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
import numpy as np

from mo.front.common.replacement import FrontReplacementPattern
from mo.front.tf.graph_utils import create_op_with_const_inputs
from mo.graph.graph import Graph, rename_node
from mo.ops.clamp import Clamp


class AttributedClampNormalizer(FrontReplacementPattern):
    """
    This transformation converts AttributedClamp operation (min/max are specified as attribute) to Clamp
    operation.
    """
    enabled = True

    def find_and_replace_pattern(self, graph: Graph):
        for attr_clamp in graph.get_op_nodes(op='AttributedClamp'):
            original_name = attr_clamp.soft_get('name', attr_clamp.id)

            rename_node(attr_clamp, original_name + '/TBR')
            min_value = attr_clamp.soft_get('min', np.finfo(np.float32).min)
            max_value = attr_clamp.soft_get('max', np.finfo(np.float32).max)
            new_clamp = create_op_with_const_inputs(graph, Clamp,
                                                    {1: np.array(min_value, dtype=np.float32),
                                                     2: np.array(max_value, dtype=np.float32)},
                                                    {'name': original_name})
            rename_node(new_clamp, original_name)

            attr_clamp.in_port(0).get_connection().set_destination(new_clamp.in_port(0))
            attr_clamp.out_port(0).get_connection().set_source(new_clamp.out_port(0))
            graph.remove_node(attr_clamp.id)
