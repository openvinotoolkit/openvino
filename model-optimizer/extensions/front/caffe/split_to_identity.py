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

from mo.front.common.replacement import FrontReplacementOp
from mo.graph.graph import Graph


class SplitToIdentity(FrontReplacementOp):
    """
    The Split layer in Caffe copies input blob to a number of output layers. The Split layer in Inference Engine divides
    the input blob into several peaces. The Caffe Split layer is redundant because Inference Engine takes care of
    creation of the intermediate blobs if it is necessary.

    The replacer changes the 'op' attribute of the node to 'Identity' and set all 'out' edge attributes to be 0. So the
    Identity operations are removed further in the pipeline.
    """
    op = "Split"
    enabled = True

    def replace_sub_graph(self, graph: Graph, match: dict):
        split_node = match['op']
        split_node.op = 'Identity'
        for u, v, edge_attrs in split_node.graph.out_edges(split_node.id, data=True):
            edge_attrs['out'] = 0
