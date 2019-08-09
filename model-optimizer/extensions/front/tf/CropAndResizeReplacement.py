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

from mo.front.common.partial_infer.utils import int64_array
from mo.front.common.replacement import FrontReplacementOp
from mo.front.tf.graph_utils import add_convolution_to_swap_xy_coordinates, create_op_node_with_second_input
from mo.graph.graph import Node, Graph
from mo.ops.concat import Concat
from mo.ops.reshape import Reshape
from mo.ops.unsqueeze import Unsqueeze


class CropAndResizeReplacement(FrontReplacementOp):
    """
    The CropAndResize operation from TF gets separate input with boxes coordinates and image batch indices. But
    ROIPooling operation in the Inference Engine receives them as a single concatenated input. This replacer
    concatenates two inputs into a new one.
    """
    op = "CropAndResize"
    enabled = True

    def nodes_to_remove(self, graph: Graph, match: dict):
        # do not remove matched node
        return []

    def replace_op(self, graph: Graph, node: Node):
        if node.has_and_set('inputs_preprocessed'):
            log.debug('Node "{}" has already been preprocessed'.format(node.soft_get('name')))
            return []
        # reshape tensor with batch indices to 2d
        unsqueeze_node = create_op_node_with_second_input(graph, Unsqueeze, int64_array([1]),
                                                         {'name': node.name + '/Unsqueeze'}, node.in_node(2))

        concat_op = Concat(graph, {'axis': 1, 'name': node.name + '/concat_batch_indices_and_boxes',
                                   'in_ports_count': 2})
        concat_node = concat_op.create_node([unsqueeze_node, node.in_node(1)])

        # do not remove edge with crop_size because it is needed in the partial infer
        graph.remove_edge(node.in_node(1).id, node.id)

        # input to the CropAndResize contains boxes coordinates in YXYX layout. But IE layer ROIPooling expects
        # coordinates in the XYXY layout, so convolution is added here to swap coordinates
        swapped_box_coordinates_node = add_convolution_to_swap_xy_coordinates(graph, concat_node, 5)

        # reshape locations tensor to 2D so it could be passed to Eltwise which will be converted to ScaleShift
        reshape_2d_node = create_op_node_with_second_input(graph, Reshape, int64_array([-1, 5]),
                                                           dict(name=swapped_box_coordinates_node.id + '/reshape_2d_'),
                                                           swapped_box_coordinates_node)
        graph.create_edge(reshape_2d_node, node, 0, 1)

        # do not replace any output edge
        return []
