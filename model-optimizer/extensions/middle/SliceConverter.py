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

import numpy as np

from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Graph, Node
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.const import Const
from mo.ops.crop import Crop
from mo.ops.strided_slice import StridedSlice
from mo.utils.error import Error


def convert_negative_indices(indices: np.array, shape: np.array):
    for ind, value in enumerate(indices):
        if value < 0:
            indices[ind] += shape[ind]


class ConvertSlice(MiddleReplacementPattern):
    """
    This class convert Slice operation to Crop, Split or StridedSlice depends on parameters
    """

    enabled = True
    op = "Slice"
    force_clean_up = True

    def run_after(self):
        from extensions.middle.pass_separator import MiddleStart
        return [MiddleStart]

    def pattern(self):
        return dict(
            nodes=[
                ('slice', dict(kind='op', op='Slice'))
            ],
            edges=[]
        )

    @staticmethod
    def convert_onnx_slice_opset10(node: Node):
        """
        Converts the Slice node from ONNX opset10 to StridedSlice.
        :param node: Slice node
        :return: None
        """
        graph = node.graph

        input_shape = node.in_port(0).data.get_shape()
        output_shape = node.out_port(0).data.get_shape()
        starts = node.in_port(1).data.get_value()
        ends = node.in_port(2).data.get_value()
        if starts is None or ends is None:
            raise Error('The input with starts or end is not constant for node {}'.format(node.id))

        # in ONNX the value for 'ends' is usually -1 which is translated to maximum possible value of int64. This
        # value must be converted to maximum of int32 because such big values do not fit into the int32 which is
        # supported by the StridedSlice layer
        ends = int64_array([np.iinfo(np.int32).max if item > np.iinfo(np.int32).max else item for item in ends])
        if node.is_in_port_connected(3):
            axes = node.in_port(3).data.get_value()
            if axes is None:
                raise Error('The input with axes is not constant for node {}'.format(node.id))
        else:
            axes = int64_array(list(range(starts.size)))

        if node.is_in_port_connected(4):
            steps = node.in_port(4).data.get_value()
            if steps is None:
                raise Error('The input with steps is not constant for node {}'.format(node.id))
        else:
            steps = np.ones([starts.size])

        ss_begin_mask = np.zeros(len(input_shape), dtype=np.int32)
        ss_end_mask = np.zeros(len(input_shape), dtype=np.int32)
        ss_begin = np.zeros(len(input_shape), dtype=np.int32)
        ss_end = np.zeros(len(input_shape), dtype=np.int32)
        ss_steps = np.ones(len(input_shape), dtype=np.int32)

        # prepare inputs and attributes for the StridedSlice layer
        for i, axis in enumerate(axes):
            if starts[i] != 0:
                ss_begin_mask[axis] = 1
                ss_begin[axis] = starts[i]

            ss_end_mask[axis] = 1
            ss_end[axis] = ends[i]

            ss_steps[axis] = steps[i]

        begin_node = Const(graph, {'value': ss_begin, 'force_precision': 'I32'}).create_node()
        end_node = Const(graph, {'value': ss_end, 'force_precision': 'I32'}).create_node()
        strides_node = Const(graph, {'value': ss_steps, 'force_precision': 'I32'}).create_node()

        ss = StridedSlice(graph, dict(new_axis_mask=np.zeros(len(output_shape), dtype=np.int32),
                                      shrink_axis_mask=np.zeros(len(output_shape), dtype=np.int32),
                                      ellipsis_mask=np.zeros(len(output_shape), dtype=np.int32),
                                      begin_mask=ss_begin_mask,
                                      end_mask=ss_end_mask)).create_node()
        node.in_port(0).get_connection().set_destination(ss.in_port(0))
        begin_node.out_port(0).connect(ss.in_port(1))
        end_node.out_port(0).connect(ss.in_port(2))
        strides_node.out_port(0).connect(ss.in_port(3))
        node.out_port(0).get_connection().set_source(ss.out_port(0))

    def replace_pattern(self, graph: Graph, match: dict):
        node = match['slice']

        input = node.in_node(0)
        output_data = node.out_node()

        # ONNX 10 opset case
        if len(node.in_nodes()) >= 3 and node.has_valid('format') and node['format'] == 'onnx':
            self.convert_onnx_slice_opset10(node)
            return

        # Caffe case
        if not node.has_valid('start') or not node.has_valid('end'):
            return

        begin = node.start
        end = node.end
        axis = node.axis if node.has_valid('axis') else np.arange(begin.size)

        # Check whether operation use only one axis or not
        axes_begin = np.zeros(len(input.shape), dtype=np.int32)
        axes_end = np.zeros(len(input.shape), dtype=np.int32)
        ss_begin = np.zeros(len(input.shape), dtype=np.int32)
        ss_end = np.zeros(len(input.shape), dtype=np.int32)
        dims = 0
        axes = np.zeros(begin.size)
        for i in range(len(axis)):
            if begin[i] != 0 or end[i] < input.shape[axis[i]]:
                dims += 1
                axes[i] = 1
                if begin[i] != 0:
                    axes_begin[axis[i]] = 1
                    ss_begin[axis[i]] = begin[i]
                if end[i] < input.shape[axis[i]]:
                    axes_end[axis[i]] = 1
                    ss_end[axis[i]] = end[i]
        axes = np.array(axes, dtype=bool)

        if dims == 1 or dims == 0:
            # If Slice use only one axis or no axis, than
            # convert Slice to StridedSlice
            ss = StridedSlice(graph, dict(new_axis_mask=np.zeros(len(output_data.shape), dtype=np.int32),
                                          shrink_axis_mask=np.zeros(len(output_data.shape), dtype=np.int32),
                                          ellipsis_mask=np.zeros(len(output_data.shape), dtype=np.int32),
                                          begin_mask=axes_begin,
                                          end_mask=axes_end))

            convert_negative_indices(ss_begin, input.shape)
            convert_negative_indices(ss_end, input.shape)

            begin_node = Const(graph, {'value': ss_begin, 'force_precision': 'I32'}).create_node_with_data()
            end_node = Const(graph, {'value': ss_end, 'force_precision': 'I32'}).create_node_with_data()

            ss.create_node_with_data(inputs=[input, begin_node, end_node], data_nodes=[output_data])
            # Remove unnecessary edges from and to to Slice vertex
            graph.remove_edge(input.id, node.id)
            graph.remove_edge(node.id, output_data.id)
        else:
            # If Slice use more than one axis use Crop layer
            crop = Crop(graph, dict(axis=axis[axes],
                                    offset=begin[axes]))
            # creating node with data
            crop.create_node_with_data(inputs=[input], data_nodes=[output_data])

            # Remove unnecessary edges from and to to Slice vertex
            graph.remove_edge(input.id, node.id)
            graph.remove_edge(node.id, output_data.id)
