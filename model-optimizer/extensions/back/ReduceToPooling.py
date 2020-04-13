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

import numpy as np

from extensions.back.AvgPool import AvgPool
from extensions.back.MaxPool import MaxPool
from extensions.back.ScalarConstNormalize import ScalarNormalize
from extensions.ops.ReduceOps import reduce_map
from mo.back.replacement import BackReplacementPattern
from mo.front.caffe.extractors.utils import get_canonical_axis_index
from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Graph
from mo.ops.concat import Concat
from mo.ops.const import Const
from mo.ops.pooling import Pooling
from mo.ops.power import AttributedPower
from mo.ops.reshape import Reshape


class ReduceMerge(BackReplacementPattern):
    """
    Fuses sequence of Reduces of the same type into one Reduce layer of this particular type with updated axes input
    Limitations:
        - `keep_dims` attribute should be the same for all Reduces in the sequence
        - in case `keep_dims`=False: next Reduce axes should be strictly less than previous Reduce axes
    """
    enabled = True
    force_clean_up = True

    def run_before(self):
        return [ReduceReplacer, ScalarNormalize]

    @staticmethod
    def fuse_reduces(first_reduce, second_reduce):
        first_reduce_name = first_reduce.soft_get('name', first_reduce.id)
        second_reduce_name = second_reduce.soft_get('name', second_reduce.id)
        reduce_type = first_reduce.type

        assert first_reduce.type == second_reduce.type

        if len(first_reduce.out_port(0).get_destinations()) != 1:
            # data dependency
            return

        if first_reduce.keep_dims != second_reduce.keep_dims:
            return

        first_axes = first_reduce.in_port(1).data.get_value()
        second_axes = second_reduce.in_port(1).data.get_value()
        if first_axes is None or second_axes is None:
            # dynamic axes merging is not supported
            return

        if not first_reduce.keep_dims:
            if not np.all(first_axes > second_axes):
                # indexing of upper reduce input dimensions changed
                return

        graph = second_reduce.graph

        new_axes = Concat(graph, {'name': second_reduce_name + '/Axes', 'axis': int64_array(0), 'in_ports_count': 2,
                                  'override_output_shape': True}).create_node()
        new_axes.in_port(0).connect(first_reduce.in_port(1).get_source())
        new_axes.in_port(1).connect(second_reduce.in_port(1).get_source())

        first_reduce.in_port(0).get_source().node['need_shape_inference'] = True
        first_reduce.in_port(0).get_source().node['override_output_shape'] = True

        second_reduce.in_port(1).get_connection().set_source(new_axes.out_port(0))

        first_reduce.out_port(0).get_connection().set_source(first_reduce.in_port(0).get_connection().get_source())
        first_reduce.in_port(1).disconnect()
        graph.remove_node(first_reduce.id)

        log.debug('{0} nodes {1} and {2} were fused to a single {2} node with updated axes input'
                  ''.format(reduce_type, first_reduce_name, second_reduce_name))

    def find_and_replace_pattern(self, graph: Graph):
        rsorted_nodes = graph.pseudo_topological_sort(reverse=True)
        for reduce_type in reduce_map.keys():
            reduces_of_type = [n for n in rsorted_nodes if n.id in graph and n.soft_get('type') == reduce_type]
            for second_reduce_node in reduces_of_type:
                if second_reduce_node.id not in graph:
                    continue
                first_reduce_node = second_reduce_node.in_port(0).get_source().node
                if first_reduce_node.soft_get('type', None) == reduce_type:
                    ReduceMerge.fuse_reduces(first_reduce=first_reduce_node, second_reduce=second_reduce_node)


class ReduceReplacer(BackReplacementPattern):
    enabled = True
    graph_condition = [lambda graph: not graph.graph['cmd_params'].generate_experimental_IR_V10]

    force_clean_up = True

    pool_method_map = {
        'ReduceMax': 'max',
        'ReduceMean': 'avg',
        'ReduceSum': 'avg',
    }

    supported_reduce_types = pool_method_map.keys()

    def run_before(self):
        from extensions.back.ReshapeMutation import ReshapeMutation
        return [ReshapeMutation, MaxPool, AvgPool, ScalarNormalize]

    def pattern(self):
        return dict(
            nodes=[
                ('reduce', dict(kind='op', type=lambda node_type: node_type in self.supported_reduce_types))
            ],
            edges=[]
        )

    @staticmethod
    def initial_reshape_dim_normalizer(number_of_elements: np.int64):
        """
        decomposes `number_of_elements` into the product of two numbers with minimum distance
        """
        new_window = [1, number_of_elements]
        for divisor in range(2, int(np.sqrt(number_of_elements)) + 1):
            if number_of_elements % divisor == 0:
                quotient = number_of_elements / divisor
                if abs(quotient - divisor) < abs(new_window[0] - new_window[1]):
                    new_window = [int(quotient), int(divisor)]

        assert np.prod(new_window) == number_of_elements
        return sorted(new_window)

    def replace_pattern(self, graph: Graph, match: dict):
        node = match['reduce']

        if node.out_port(0).data.get_value() is not None:
            # We leave Reduce* operations located in constant sub-graph as is
            # to keep model reshapable with --keep_shape_ops cli key
            return

        reduce_type = node.type
        if reduce_type not in self.pool_method_map:
            log.error("Reduce type {} is not included in pool_method_map. Please update pool_method_map with new key "
                      "{}".format(reduce_type, reduce_type))
            return

        input_data = node.in_node()
        output_data = node.out_node()

        input_shape = node.in_port(0).data.get_shape()
        output_shape = node.out_port(0).data.get_shape()

        # normalize node axes to exclude negative indices
        axes_data_value = node.in_port(1).data.get_value()
        axes = int64_array([axes_data_value.item()]) if axes_data_value.size == 1 else axes_data_value
        axes = [get_canonical_axis_index(input_shape, a) for a in axes]
        axes = sorted(axes)

        # Check that values in axes list are consecutive
        for idx in range(1, len(axes)):
            if axes[idx] != (axes[idx - 1] + 1):
                log.error("Reduce with not consecutive axes {} is not supported ".format(axes))
                return
        # So now we are sure that we can convert Reduce to appropriate operation

        # 1. Calculate shape that will be used in reduction
        reduction_dim = np.prod([input_shape[idx] for idx in axes])
        begin_dims = np.array([input_shape[idx] for idx in range(axes[0])])
        end_dim = np.prod([input_shape[idx] for idx in range(axes[-1] + 1, len(input_shape))])

        # 2. Create reshape with appropriate shape
        if len(begin_dims) > 2:
            if 0 not in axes:
                begin_dims = int64_array([begin_dims[0], np.prod(begin_dims[1:])])
            else:
                begin_dims = int64_array([np.prod(begin_dims[0:-1]), begin_dims[-1]])
        else:
            # Expand begin_dims to 2
            begin_dims = int64_array(np.append(begin_dims, [1] * (2 - len(begin_dims))))

        reshape_shape = int64_array([*begin_dims, reduction_dim, end_dim])
        pool_window = int64_array([1, 1, reduction_dim, 1])

        if end_dim == 1:
            new_window = ReduceReplacer.initial_reshape_dim_normalizer(reduction_dim)
            reshape_shape = int64_array([*begin_dims, *new_window])
            pool_window = int64_array([1, 1, *new_window])

        # 3. Reduce => Reshape->Pooling->Reshape
        reshape_op = Reshape(graph, {'name': node.id + '/Reshape'})
        reshape_dim_const_data = Const(graph, {'name': node.id + '/Reshape/Dim',
                                               'value': reshape_shape}).create_node_with_data()

        final_reshape_op = Reshape(graph, {'name': node.id + '/FinalReshape'})
        final_reshape_dim_const_data = Const(graph, {'name': node.id + '/FinalReshape/Dim',
                                                     'value': output_shape}).create_node_with_data()
        pooling_op = Pooling(graph,
                             dict(name=node.id + '/Pool',
                                  window=pool_window,
                                  output_spatial_shape=None,
                                  batch_dims=int64_array([0]),
                                  channel_dims=int64_array([1]),
                                  exclude_pad='false', pool_method=self.pool_method_map[reduce_type]))

        graph.remove_edge(input_data.id, node.id)
        graph.remove_edge(node.id, output_data.id)

        if np.array_equal(input_shape, reshape_shape):
            input_to_pooling = input_data
        else:
            input_to_pooling = reshape_op.create_node_with_data(inputs=[input_data, reshape_dim_const_data])
        pooling = pooling_op.create_node_with_data(inputs=[input_to_pooling])
        final_reshape_op.create_node_with_data(inputs=[pooling, final_reshape_dim_const_data], data_nodes=output_data)

        # convert batch dimension to 0 to produce reshape-able IR over the batch dimension
        if 0 not in axes:
            reshape_dim_const_data.in_node(0).value[0] = 0
            final_reshape_dim_const_data.in_node(0).value[0] = 0

        # 4. If it is reduction with summation, we need to multiply by size of the reduction slice with Mul op
        if reduce_type == 'ReduceSum':
            output_data.in_node().insert_node_with_data_after(
                output_data,
                AttributedPower,
                {'name': node.name + '/Mul', 'scale': float(reduction_dim)}
            )


class ReduceLogicalReplacer(BackReplacementPattern):
    enabled = True
    graph_condition = [lambda graph: not graph.graph['cmd_params'].generate_experimental_IR_V10]

    def pattern(self):
        return dict(
            nodes=[
                ('reduce', dict(kind='op', type=lambda node_type: node_type is not None and
                                                                  node_type.startswith('ReduceLogical')))
            ],
            edges=[]
        )

    def replace_pattern(self, graph: Graph, match: dict):
        node = match['reduce']
        node.type = node.type.replace('Logical', '')
