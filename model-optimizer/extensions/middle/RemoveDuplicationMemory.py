"""
 Copyright (c) 2019 Intel Corporation

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

from mo.graph.graph import Graph, Node
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.crop import Crop


class RemoveMemoryDuplicationPattern(MiddleReplacementPattern):
    """
    Remove Splice nodes with context that is included in context of another Splice with the same input 
    """
    enabled = False

    @staticmethod
    def pattern():
        return dict(
            nodes=[('op', dict(op='Splice'))],
            edges=[])

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        if len(match['op'].in_nodes()) == 0:
            return

        mem = match['op']
        in_mem = mem.in_node(0)
        context = mem['context']
        outs = in_mem.out_nodes()

        for out in outs:
            if out['op'] == 'Splice' and out.id != mem.id and set(out['context']).issubset(set(context)):
                left_cont_out = out['context'][0]
                left_cont = context[0]

                out_node = out.out_node()
                for out_name, out_edge in out_node.get_outputs():
                    out_transfer = Node(graph, out_name)
                    if out_transfer['op'] == 'Crop':
                        # modify existing Crop to get right data from larger Splice
                        out_transfer['offset'] = out_transfer['offset'] + (left_cont_out - left_cont) * in_mem.shape[-1]
                    else:
                        # insert Crop if we have not one
                        out_transfer.in_port(out_edge['in']).disconnect()
                        crop_node = Crop(graph, {'name': graph.unique_id(prefix='Splice_crop_'),
                                                 'offset': (left_cont_out - left_cont) * in_mem.shape[-1],
                                                 'dim': np.array([len(out['context']) * in_mem.shape[-1]]),
                                                 'axis': np.array([-1])}).create_node()
                        out.out_port(0).connect(crop_node.in_port(0))
                        crop_node.out_port(0).connect(out_transfer.in_port(out_edge['in']))
                        crop_node.out_node(0).shape = out_node.shape

                        out_transfer = crop_node

                    # move edge from old Splice to larger
                    in_port = graph.get_edge_data(out_node.id, out_transfer.id)[0]['in']
                    out_transfer.in_port(0).disconnect()
                    mem.out_port(0).connect(out_transfer.in_port(in_port))

                graph.remove_node(out.id)
