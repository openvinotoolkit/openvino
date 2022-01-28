# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import int64_array, mo_array
from openvino.tools.mo.front.common.replacement import FrontReplacementOp
from openvino.tools.mo.front.tf.graph_utils import create_op_with_const_inputs
from openvino.tools.mo.graph.graph import Graph, rename_nodes
from openvino.tools.mo.ops.Cast import Cast
from openvino.tools.mo.ops.ReduceOps import ReduceProd
from openvino.tools.mo.ops.elementwise import Add, Div, Mul
from openvino.tools.mo.ops.gather import Gather
from openvino.tools.mo.ops.range import Range
from openvino.tools.mo.ops.reshape import Reshape
from openvino.tools.mo.ops.shape import Shape
from openvino.tools.mo.ops.slice import Slice
from openvino.tools.mo.ops.squeeze import Squeeze
from openvino.tools.mo.ops.tile import Tile
from openvino.tools.mo.utils.error import Error


class ArangeLikeReplacer(FrontReplacementOp):
    op = 'arange_like'
    enabled = True

    def replace_sub_graph(self, graph: Graph, match: dict):
        node = match['op']
        name = node.soft_get('name', node.id)
        axis = node.axis
        input_shape_node = Shape(graph, {'name': name + '/ShapeOf'}).create_node()
        range_node = create_op_with_const_inputs(graph, Range, {0: mo_array(node.start),
                                                                2: mo_array(node.step)}, {'name': name + '/Range'})
        node.in_port(0).get_connection().set_destination(input_shape_node.in_port(0))

        if axis is not None:
            '''
            Replace arange_like op to subgraph:
            Shape - Gather - Range
            '''
            gather_node = create_op_with_const_inputs(graph, Gather, {1: int64_array([axis]),
                                                                      2: int64_array(0)},
                                                      {'name': name + '/Gather'})
            input_shape_node.out_port(0).connect(gather_node.in_port(0))
            gather_node.out_port(0).connect(range_node.in_port(1))
            node.out_port(0).get_connection().set_source(range_node.out_port(0))
            rename_nodes([(node, name + '/ShouldBeDeleted'), (range_node, name)])
        else:
            r'''
            Replace arange_like op to subgraph:
                    |
                 ShapeOf ----------- | 
                    |                |
                 ReduceProd          |
                    |                |
                  Range              |
                    |                |
                 Reshape ----------- | 
                    |
            '''

            flattened_shape_node = create_op_with_const_inputs(graph, ReduceProd, {1: int64_array([0])},
                                                               {'name': input_shape_node.name + '/ReduceProd',
                                                                'keep_dims': True})
            reshape_backward_node = Reshape(graph, {'name': name + '/Reshape_backward'}).create_node()

            input_shape_node.out_port(0).connect(flattened_shape_node.in_port(0))
            flattened_shape_node.out_port(0).connect(range_node.in_port(1))
            range_node.out_port(0).connect(reshape_backward_node.in_port(0))
            input_shape_node.out_port(0).connect(reshape_backward_node.in_port(1))
            node.out_port(0).get_connection().set_source(reshape_backward_node.out_port(0))
            rename_nodes([(node, name + '/ShouldBeDeleted'), (reshape_backward_node, name)])

        if node.repeat != 1:
            r"""
            First, we generate the correct stop value for Range like new_stop_value = stop_value // repeat + 1.
            Then repeats each value of the interval using Tile. After that we can get a longer interval
            so we reduce it with Slice.
            
            Sub-graph after Range node will be look like
            
            Range - Reshape([-1, 1]) - Tile([1, repeat]) - Reshape(-1) - Slice
            
            """

            if node.repeat < 1:
                raise Error("Unexpected value {} of the attribute 'repeat' for the node {}". format(node.repeat, name))

            div_node = create_op_with_const_inputs(graph, Div, {1: int64_array([node.repeat])},
                                                   {'name': name + '/Divide'})
            add_node = create_op_with_const_inputs(graph, Add, {1: int64_array([1])},
                                                   {'name': div_node.name + '/Add'})
            cast_node = Cast(graph, {'name': name + '/ConvertToI64', 'dst_type': np.int64}).create_node()

            cast_node.out_port(0).connect(div_node.in_port(0))
            div_node.out_port(0).connect(add_node.in_port(0))
            range_node.in_port(1).get_connection().set_destination(cast_node.in_port(0))
            add_node.out_port(0).connect(range_node.in_port(1))

            tile_forward_reshape = create_op_with_const_inputs(graph, Reshape, {1: int64_array([-1, 1])},
                                                               {'name': range_node.name + '/ForwardReshape'})
            tile = create_op_with_const_inputs(graph, Tile, {1: int64_array([1, node.repeat])},
                                               {'name': tile_forward_reshape.name + '/Tile'})
            tile_backward_reshape = create_op_with_const_inputs(graph, Reshape, {1: int64_array([-1])},
                                                                {'name': tile.name + '/BackwardReshape'})
            slice_node = create_op_with_const_inputs(graph, Slice, {1: int64_array([0]), 3: int64_array([0]),
                                                                    4: int64_array([1])},
                                                     {'name': tile_backward_reshape.name + '/Slice'})

            tile_forward_reshape.out_port(0).connect(tile.in_port(0))
            tile.out_port(0).connect(tile_backward_reshape.in_port(0))
            tile_backward_reshape.out_port(0).connect(slice_node.in_port(0))
            slice_node.in_port(2).connect(div_node.in_port(0).get_source())

            range_node.out_port(0).get_connection().set_source(slice_node.out_port(0))
            range_node.out_port(0).connect(tile_forward_reshape.in_port(0))

            if axis is not None:
                rename_nodes([(range_node, name + '/Range'), (slice_node, name)])

        # MXNet arange_like op has no stop attribute and the result tensor always matches the input shape, so
        # we have to correct the stop value for the Range node if step != 1 or start != 0
        if node.step != 1:
            # If step attribute is not integer, we will generate an interval with a larger size and then reduce it
            # using Slice
            true_elements_count_port = range_node.in_port(1).get_source()
            mul_value = np.ceil(node.step) if node.step > 0 else np.floor(node.step)
            stop_value = create_op_with_const_inputs(graph, Mul, port_value_dict={1: mo_array(np.ceil(mul_value))},
                                                     op_attrs={'name': range_node.name + '/Stop'})
            range_node.in_port(1).get_connection().insert_node(stop_value)

            slice_range_values = create_op_with_const_inputs(graph, Slice, {1: int64_array([0]), 3: int64_array([0]),
                                                                            4: int64_array([1])},
                                                             {'name': range_node.name + '/Slice'})
            slice_range_values.in_port(2).connect(true_elements_count_port)
            range_node.out_port(0).get_connection().insert_node(slice_range_values)

            if axis is not None and node.repeat == 1:
                rename_nodes([(range_node, name + '/Range'), (slice_range_values, name)])

        if node.start != 0:
            correct_stop_value = create_op_with_const_inputs(graph, Add, port_value_dict={1: mo_array(node.start)},
                                                             op_attrs={'name': range_node.name + '/Correct_Stop'})
            range_node.in_port(1).get_connection().insert_node(correct_stop_value)

        # Range node supports only scalar inputs
        squeeze_node = create_op_with_const_inputs(graph, Squeeze, port_value_dict={1: int64_array(0)},
                                                   op_attrs={"name": range_node.name + '/Stop/Squeeze'})
        range_node.in_port(1).get_connection().insert_node(squeeze_node)
