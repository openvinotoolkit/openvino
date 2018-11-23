"""
 Copyright (c) 2018 Intel Corporation

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
import networkx as nx
import numpy as np

from copy import deepcopy

from extensions.middle.AddReshapeAfterStridedSlice import AddReshapeAfterStridedSlice
from extensions.middle.FusePermutesSequence import FusePermutesSequence
from extensions.middle.ShufflenetReshape import ReshapeSoftmaxReshape
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.op import Op
from mo.ops.permute import Permute


class PixelLinkReshape(MiddleReplacementPattern):
    """
      Transform adds Permutes around Reshapes that pack 4 dimensions in 2, than 
      do Softmax and then unpack it back to 5 dims. 
    """
    enabled = True

    def run_before(self):
        return [FusePermutesSequence, ReshapeSoftmaxReshape, AddReshapeAfterStridedSlice]

    def run_after(self):
        return []

    def pattern(self):
        return dict(nodes=[('reshape_split', dict(kind='op', type='Reshape')),
                           ('reshape_split_data', dict(kind='data')),
                           ('reshape_pack', dict(kind='op', type='Reshape')),
                           ('reshape_data', dict(kind='data')),
                           ('softmax', dict(kind='op', type='SoftMax')),
                           ('softmax_data', dict(kind='data')),
                           ('reshape_unpack', dict(kind='op', type='Reshape')),
                           ('reshape_unpack_data', dict(kind='data')),
                           ('strided_slice', dict(kind='op', op='StridedSlice')),
                         ],
                    edges=[('reshape_split', 'reshape_split_data'),
                           ('reshape_split_data', 'reshape_pack'),
                           ('reshape_pack', 'reshape_data'),
                           ('reshape_data', 'softmax'),
                           ('softmax', 'softmax_data'),
                           ('softmax_data', 'reshape_unpack'),
                           ('reshape_unpack', 'reshape_unpack_data'),
                           ('reshape_unpack_data', 'strided_slice')])

    def is_reshape_bad(self, node_pack, node_unpack, node_ss):
        shape_in = node_pack.in_node(0).shape
        shape_out = node_pack.out_node(0).shape

        if len(shape_in) == 5 and len(shape_out) == 2:
            shape_in = node_unpack.in_node(0).shape
            shape_out = node_unpack.out_node(0).shape
            if len(shape_out) == 5 and len(shape_in) == 2:
                if node_ss.slices[1].stop == shape_out[1] and node_ss.slices[1].start == 0 and node_ss.slices[
                    1].step == 1 and \
                        node_ss.slices[2].stop == shape_out[2] and node_ss.slices[2].start == 0 and node_ss.slices[
                    2].step == 1 and \
                        node_ss.slices[3].stop == shape_out[3] and node_ss.slices[3].start == 0 and node_ss.slices[
                    3].step == 1 and \
                        node_ss.shrink_axis_mask[4] and \
                        not node_ss.shrink_axis_mask[1] and not node_ss.shrink_axis_mask[2] and not \
                        node_ss.shrink_axis_mask[3]:
                    return True
                else:
                    return False
        else:
            return False

    def replace_pattern(self, graph: nx.MultiDiGraph, match: dict):
        if graph.graph['layout'] != 'NHWC':
            return

        if self.is_reshape_bad(match['reshape_pack'], match['reshape_unpack'], match['strided_slice']):
            log.info("Reshape that pack/unpack several dimensions detected {}".format(match['reshape_pack'].id))
            node_split = match['reshape_split']

            # insert Permute before reshape
            data_node = Op._create_data_node(graph, node_split.name + "/Permute_before_data")
            permute_before = Permute(graph, dict(name=node_split.name + "/Permute_before",
                                                 order=np.array([0, 2, 3, 1])))
            in_node = node_split.in_node(0)
            attrs = deepcopy(graph.get_edge_data(in_node.id, node_split.id)[0])
            graph.remove_edge(in_node.id, node_split.id)
            permute_before_node = permute_before.create_node_with_data([in_node], permute_before.attrs,
                                                                       data_nodes=[data_node])
            graph.add_edge(permute_before_node.id, node_split.id, **attrs)

            node = match['reshape_pack']
            new_reshape_shape = np.concatenate((np.array([node.in_node(0).shape[0]]),
                                                np.array([np.prod(node.in_node(0).shape[[1, 2, 3]])]),
                                                np.array([node.in_node(0).shape[-1]])))

            node.dim = new_reshape_shape

            # insert Permute after reshape
            data_node = Op._create_data_node(graph, node.name + "/Permute_after_data", {'shape': node.dim})
            permute_after = Permute(graph, dict(name=node.name + "/Permute_after",
                                                order=np.array([0, 2, 1])))
            out_node = node.out_node(0)
            out_node.shape = new_reshape_shape[np.array([0, 2, 1])]
            attrs = deepcopy(graph.get_edge_data(node.id, out_node.id)[0])
            graph.remove_edge(node.id, out_node.id)

            permute_after_node = permute_after.create_node_with_data([data_node], permute_after.attrs,
                                                                     data_nodes=[out_node])
            graph.add_edge(node.id, data_node.id, **attrs)

            # update softmax shape
            node_softmax = match['softmax']
            node_softmax.out_node(0).shape = out_node.shape

            # revert strided slice and reshape
            node_ss = match['strided_slice']
            node_unpack = match['reshape_unpack']

            unpack_out = node_unpack.out_node(0).id
            ss_out = node_ss.out_node(0).id

            #gather edge attributes
            soft_reshape_attrs = deepcopy(graph.get_edge_data(node_softmax.out_node(0).id, node_unpack.id)[0])
            reshape_data_attrs = deepcopy(graph.get_edge_data(node_unpack.id, unpack_out)[0])
            reshape_ss_attrs = deepcopy(graph.get_edge_data(unpack_out, node_ss.id)[0])
            ss_out_attrs = deepcopy(graph.get_edge_data(node_ss.id, ss_out)[0])

            #remove all edges in Softmax->Reshape->StridedSlice chain
            graph.remove_edge(node_softmax.out_node(0).id, node_unpack.id)
            graph.remove_edge(node_unpack.id, unpack_out)
            graph.remove_edge(unpack_out, node_ss.id)
            graph.remove_edge(node_ss.id, ss_out)

            #add new edges to get chain Softmax->StridedSlice->Reshape
            graph.add_edge(node_softmax.out_node(0).id, node_ss.id, **soft_reshape_attrs)
            graph.add_edge(node_ss.id, unpack_out, **reshape_data_attrs)
            graph.add_edge(unpack_out, node_unpack.id, **reshape_ss_attrs)
            graph.add_edge(node_unpack.id, ss_out, **ss_out_attrs)

            #update output shape and parameters for StridedSlice
            node_ss.out_node(0).shape = np.zeros(3)
            node_ss.out_node(0).shape[0] = out_node.shape[0]
            node_ss.out_node(0).shape[1] = 1
            node_ss.out_node(0).shape[2] = out_node.shape[2]

            old_slices = node_ss.slices.copy()
            node_ss.slices = []
            node_ss.slices.append(old_slices[0])
            node_ss.slices.append(old_slices[-1])
            node_ss.slices.append(slice(0, out_node.shape[2], 1))
            node_ss.shrink_axis_mask = [False, False, False]
            node_ss.new_axis_mask = [False, False, False]

            #update Reshape attribute
            node_unpack.dim = np.delete(node_unpack.dim, 4)
            #prevent permute for reshape because it gives wrong result
            node_unpack['nchw_layout'] = True
            node_unpack.out_node(0)['nchw_layout'] = True
