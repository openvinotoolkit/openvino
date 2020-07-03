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
import numpy as np

from mo.graph.graph import Graph
from mo.middle.replacement import MiddleReplacementPattern


class MxNetSqueezeNormalizer(MiddleReplacementPattern):
    """
    MxNet squeeze keeps at least 1D output of Squeeze operation which is different from
    """
    enabled = True

    def find_and_replace_pattern(self, graph: Graph):
        for node in graph.get_op_nodes(type='Squeeze', keep_at_least_1d=True):
            name = node.soft_get('name', node.id)
            output_shape = node.out_port(0).data.get_shape()
            assert output_shape is not None, 'Squeeze node {} output shape is not set'.format(name)

            del node['keep_at_least_1d']
            node.infer(node)
            if np.array(node.out_port(0).data.get_shape(), []):
                if node.is_in_port_connected(1):
                    # keep one axes non squeezed
                    pass
                else:
                    # insert unsqueeze after the node keeping the name
                    pass
