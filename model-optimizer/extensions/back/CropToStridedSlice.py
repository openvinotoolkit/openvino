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

from extensions.back.ForceStrictPrecision import ForceStrictPrecision
from extensions.ops.elementwise import Add
from mo.back.replacement import BackReplacementPattern
from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Graph, Node
from mo.ops.const import Const
from mo.ops.shape import Shape
from mo.ops.strided_slice import StridedSlice


class CropToStridedSlice(BackReplacementPattern):
    enabled = True
    force_clean_up = True

    graph_condition = [lambda graph: graph.graph['cmd_params'].generate_experimental_IR_V10]

    def run_before(self):
        return [ForceStrictPrecision]

    @staticmethod
    def pattern():
        return dict(
            nodes=[
                ('crop', dict(type='Crop'))
            ],
            edges=[]
        )

    @staticmethod
    def mask_normalizer(shape_rank: int, axes: np.ndarray, values: np.ndarray):
        mask = np.zeros(shape_rank, dtype=np.int64)
        for i, axis in enumerate(axes):
            mask[axis] = values[i]
        return mask

    @staticmethod
    def list_to_ndarray(val):
        return np.array(val) if np.array(val).ndim != 0 else np.array([val])

    def replace_pattern(self, graph: Graph, match: [str, Node]):
        node = match['crop']
        assert node.has_valid('axis')
        node.axis = self.list_to_ndarray(node.axis)

        in_shape = node.in_port(0).data.get_shape()
        shape_rank = in_shape.size
        axis_mask = int64_array([1 if i in node.axis else 0 for i in range(shape_rank)])
        begin_mask = axis_mask.copy()
        end_mask = axis_mask.copy()

        if len(node.in_nodes()) == 2 and node.has_valid('offset'):
            # Crop Type 1
            begin = Const(graph, {'value': self.mask_normalizer(shape_rank, node.axis, node.offset)}).create_node()
            shape = Shape(graph, {'name': node.name + '/shape_of_crop'}).create_node()
            end = Add(graph, {'name': node.name + '/end'}).create_node()
            node.in_port(1).get_connection().get_source().connect(shape.in_port(0))
            node.in_port(1).disconnect()
            shape.out_port(0).connect(end.in_port(0))
            begin.out_port(0).connect(end.in_port(1))
        elif node.has_valid('dim') and node.has_valid('offset'):
            # Crop Type 2
            node.dim = self.list_to_ndarray(node.dim)
            node.offset = self.list_to_ndarray(node.offset)
            assert node.dim.size == node.offset.size == node.axis.size

            begin = Const(graph, {'value': self.mask_normalizer(shape_rank, node.axis, node.offset)}).create_node()
            end_values = np.array([node.offset[i] + node.dim[i] for i in range(len(node.dim))])
            end = Const(graph, {'value': self.mask_normalizer(shape_rank, node.axis, end_values)}).create_node()
        elif node.has_valid('crop_begin') and node.has_valid('crop_end'):
            # Crop Type 3
            node.crop_begin = self.list_to_ndarray(node.crop_begin)
            node.crop_end = self.list_to_ndarray(node.crop_end)
            assert len(node.crop_begin) == len(node.crop_end) == len(node.axis)

            begin = Const(graph, {'value': self.mask_normalizer(shape_rank, node.axis, node.crop_begin)}).create_node()
            shape = Shape(graph, {'name': node.name + '/shape_of_crop'}).create_node()
            const = Const(graph,
                          {'value': -1 * self.mask_normalizer(shape_rank, node.axis, node.crop_end)}).create_node()
            end = Add(graph, {'name': node.name + '/end'}).create_node()

            node.in_port(0).get_connection().get_source().connect(shape.in_port(0))
            shape.out_port(0).connect(end.in_port(0))
            const.out_port(0).connect(end.in_port(1))

        else:
            raise Exception("Unknown type of Crop")

        source = node.in_port(0).get_connection().get_source()

        stride = Const(graph, {'value': np.ones(shape_rank, dtype=np.int64)}).create_node()
        ss = StridedSlice(graph, {'name': 'Crop_', 'begin_mask': begin_mask, 'end_mask': end_mask, 'new_axis_mask': np.array([0]),
                                  'shrink_axis_mask': np.array([0]), 'ellipsis_mask': np.array([0])}).create_node()

        source.connect(ss.in_port(0))
        begin.out_port(0).connect(ss.in_port(1))
        end.out_port(0).connect(ss.in_port(2))
        stride.out_port(0).connect(ss.in_port(3))

        node.in_port(0).disconnect()
        node.out_port(0).get_connection().set_source(ss.out_port(0))

        ss['force_precision_in_ports'] = {1: 'int64', 2: 'int64', 3: 'int64'}
