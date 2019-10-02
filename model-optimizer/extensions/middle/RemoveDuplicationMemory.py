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

from mo.graph.graph import Graph
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
        mem = match['op']
        mem_shape = mem.in_port(0).data.get_shape()
        mem_parent = mem.in_port(0).get_source()
        context = mem['context']

        for child_port in mem_parent.get_destinations():
            child = child_port.node
            # check if we find Splice containing context 'context'
            if child['op'] == 'Splice' and child.id != mem.id and set(child['context']).issubset(set(context)):
                left_cont_out = child['context'][0]
                left_cont = context[0]

                for child_of_child in child.out_port(0).get_destinations():
                    out_transfer = child_of_child.node
                    out_transfer_port = child_of_child
                    if out_transfer['op'] == 'Crop':
                        # modify existing Crop to get right data from larger Splice
                        out_transfer['offset'] = out_transfer['offset'] + (left_cont_out - left_cont) * mem_shape[-1]
                    else:
                        # insert Crop if we have not one
                        child_of_child.disconnect()
                        crop_node = Crop(graph, {'name': graph.unique_id(prefix='Splice_crop_'),
                                                 'offset': (left_cont_out - left_cont) * mem_shape[-1],
                                                 'dim': np.array([len(child['context']) * mem_shape[-1]]),
                                                 'axis': np.array([-1])}).create_node()
                        child.out_port(0).connect(crop_node.in_port(0))
                        crop_node.out_port(0).connect(child_of_child)
                        crop_node.out_port(0).data.set_shape(child.out_port(0).data.get_shape())

                        out_transfer_port = crop_node.in_port(0)

                    # move edge to child from old Splice to larger
                    out_transfer_port.disconnect()
                    mem.out_port(0).connect(out_transfer_port)

                graph.remove_node(child.id)


class MergeNeighborSplicePattern(MiddleReplacementPattern):
    """
    Merge Splices with neighbor contexts, for example: [-5, 0] and [0, 3] to context [-5, 3]
    """
    enabled = False

    @staticmethod
    def pattern():
        return dict(
            nodes=[('op', dict(op='Splice'))],
            edges=[])

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        mem = match['op']
        mem_shape = mem.in_port(0).data.get_shape()
        mem_parent = mem.in_port(0).get_source()
        context = mem['context']

        for child_port in mem_parent.get_destinations():
            child = child_port.node
            if child['op'] == 'Splice' and child.id != mem.id and \
               (child['context'][0] == context[-1] or child['context'][0] == context[-1]):

                new_context = list(context)
                new_context.extend(list(child['context']))
                new_context = list(set(new_context))
                new_context.sort()
                if child['context'][0] == context[-1]:
                    new_node = mem
                    rem_node = child
                else:
                    new_node = child
                    rem_node = mem

                # reset edges from rem_node to new_node
                for out_port_rem in rem_node.out_port(0).get_destinations():
                    out_transfer = out_port_rem.node
                    out_transfer_shape = out_port_rem.data.get_shape().copy()

                    out_port_rem.disconnect()

                    if out_transfer['op'] == 'Crop':
                        # modify existing Crop to get right data from larger Splice
                        out_transfer['offset'] = out_transfer['offset'] + (len(new_context) - len(rem_node.context)) * mem_shape[-1]
                        out_port_rem.connect(new_node.out_port(0))
                    else:
                        # insert Crop if we have not one
                        crop_node = Crop(graph, {'name': graph.unique_id(prefix='Splice_crop_'),
                                                 'offset': (len(new_context) - len(rem_node.context)) * mem_shape[-1],
                                                 'dim': np.array([len(rem_node['context']) * mem_shape[-1]]),
                                                 'axis': np.array([-1])}).create_node()
                        new_node.out_port(0).connect(crop_node.in_port(0))
                        crop_node.out_port(0).connect(out_port_rem)
                        crop_node.out_port(0).data.set_shape(out_transfer_shape)

                for out_port_rem in new_node.out_port(0).get_destinations():
                    out_transfer = out_port_rem.node
                    out_transfer_shape = out_port_rem.data.get_shape().copy()

                    if out_transfer['op'] != 'Crop':
                        # insert Crop if we have not one
                        crop_node = Crop(graph, {'name': graph.unique_id(prefix='Splice_crop_'),
                                                 'offset': np.array([0]),
                                                 'dim': np.array([len(new_node['context']) * mem_shape[-1]]),
                                                 'axis': np.array([-1])}).create_node()
                        new_node.out_port(0).connect(crop_node.in_port(0))
                        out_port_rem.disconnect()
                        crop_node.out_port(0).connect(out_port_rem)
                        crop_node.out_port(0).data.set_shape(out_transfer_shape)

                new_shape = new_node.out_port(0).data.get_shape()
                new_shape[1] += rem_node.out_port(0).data.get_shape()[1] - rem_node.in_port(0).data.get_shape()[1]
                new_node.out_port(0).data.set_shape(new_shape)
                new_node.context = new_context

                graph.remove_node(rem_node.id)
