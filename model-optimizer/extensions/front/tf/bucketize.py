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

from mo.front.common.replacement import FrontReplacementSubgraph
from mo.graph.graph import Graph
from mo.ops.const import Const


class BucketizeFrontReplacer(FrontReplacementSubgraph):
    """
    Moves the boundaries data from attribute to the second input tensor.
    """
    enabled = True

    def find_and_replace_pattern(self, graph: Graph):
        for bucketize in graph.get_op_nodes(op='Bucketize'):
            if bucketize.in_port(1).disconnected():
                assert bucketize.has_valid('boundaries'), 'The Bucketize node "{}" misses "boundaries" attribute'.format(bucketize.name)
                boundaries_node = Const(graph, {'name': bucketize.name + '/Bucketize_boundaries_', 'value': bucketize.boundaries}).create_node()
                bucketize.in_port(1).connect(boundaries_node.out_port(0))
                del bucketize['boundaries']
            else:
                log.debug('The Bucketize node input "{}" is already normalized'.format(bucketize.name))
