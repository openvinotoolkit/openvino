"""
 Copyright (c) 2017-2019 Intel Corporation

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

from extensions.ops.lstm_cell import LSTMCell
from mo.front.common.replacement import FrontReplacementSubgraph
from mo.graph.graph import Node, Graph
from mo.ops.result import Result


class BasicLSTMCell(FrontReplacementSubgraph):
    enabled = True

    # When the deprecated IR version was requested, we configure only those phases that can lead
    # to functional regressions in the version 2. BasicLSTMCell is one such transformation;
    # when it is turned off, the body of TF basic_lstm_cell is converted as-is in a decomposed form,
    # and should work in version 2.
    graph_condition = [lambda graph: graph.graph['ir_version'] != 2]

    # list of names of all original nodes that are supported by IE
    # this list is collected gradually by a separate transformation
    # original name in this case is a selected node in the pattern
    # that is returned from anchor() function
    instances_supported_by_IE = []

    def __init__(self):

        super().__init__()

        # Inputs that are required by LSTMCell operation definition
        __class__.inputs = ['input_op', 'input_hidden_state', 'input_cell_state', 'weights', 'biases']

        # Extra inputs that are not expected by LSTMCell but required for extra checks
        # at middle-end partial inference stage. They are consumed by the extended infer function
        # and then removed.
        __class__.extra_inputs = ['concat_axis', 'split_axis', 'shift_const']

        __class__.outputs = ['mul_2', 'add_1']

    def pattern(self):
        return dict(
            nodes=[
                ('concat_axis', dict()),
                ('concat', dict(op='ConcatV2')),
                ('weights', dict()),
                ('matmul', dict(op='MatMul')),
                ('biases', dict()),
                ('biasadd', dict(op='Add')),
                ('split_axis', dict()),
                ('split', dict(op='Split')),
                ('shift_const', dict()),
                ('shift', dict(op='Add')),
                ('sigmoid_0', dict(op='Sigmoid')),
                ('mul_0', dict(op='Mul')),
                ('sigmoid_1', dict(op='Sigmoid')),
                ('tanh_0', dict(op='Tanh')),
                ('mul_1', dict(op='Mul')),
                ('add_1', dict(op='Add')),
                ('tanh_1', dict(op='Tanh')),
                ('sigmoid_2', dict(op='Sigmoid')),
                ('mul_2', dict(op='Mul'))
            ],
            edges=[
                # This important block specifies how input/hidden are concatenated
                ('concat_axis', 'concat', {'in': 2}),

                ('concat', 'matmul', {'in': 0}),
                ('weights', 'matmul', {'in': 1}),
                ('matmul', 'biasadd', {'in': 0}),
                ('biases', 'biasadd', {'in': 1}),

                ('split_axis', 'split', {'in': 0}),
                ('biasadd', 'split', {'in': 1}),

                # This important block specifies how gates are ordered in TF graph
                ('split', 'sigmoid_1', {'out': 0}),  # i
                ('split', 'tanh_0', {'out': 1}),  # c
                ('split', 'shift', {'out': 2}),  # f (this is unbiased f, there is an extra addition here)
                ('split', 'sigmoid_2', {'out': 3}),  # o

                ('shift_const', 'shift', {}),
                ('shift', 'sigmoid_0', {}),
                ('sigmoid_0', 'mul_0', {}),

                ('sigmoid_1', 'mul_1', {}),
                ('tanh_0', 'mul_1', {}),

                ('mul_0', 'add_1', {}),
                ('mul_1', 'add_1', {}),

                ('add_1', 'tanh_1', {}),
                ('tanh_1', 'mul_2', {}),
                ('sigmoid_2', 'mul_2', {}),
            ])

    @staticmethod
    def anchor():
        """ Mnemonic name in the pattern that is used as an anchor name for this pattern in the original graph.
            Used for the second round of the pattern application when only a part of instances is allowed for conversion.
        """
        return 'concat'

    def replace_sub_graph(self, graph: Graph, match: dict):

        # node that is used to identify this pattern application instance for switching between supported
        # and not supported LSTMCell sub-graphs; this value will be searched in __class__.instances_supported_by_IE.
        anchor_node = match[__class__.anchor()]
        assert anchor_node.has_valid('name'), \
            'LSTMCell anchor node {} does\'t have attribute name; such nodes are not supported.'

        match['input_op'] = match['concat'].in_node(0)
        match['input_hidden_state'] = match['concat'].in_node(1)
        match['input_cell_state'] = match['mul_0'].in_node(0) \
            if match['mul_0'].in_node(0).id != match['sigmoid_0'].id else match['mul_0'].in_node(1)

        pattern_edges = self.pattern()['edges']
        pattern_edges.extend([('input_op', 'concat'), ('input_cell_state', 'mul_0'), ('input_hidden_state', 'concat')])
        inputs = graph.get_inputs_with_ports(match, pattern_edges, __class__.inputs + __class__.extra_inputs)

        lstm_op = LSTMCell(graph, dict(
            name=match['concat'].name + '/LSTMCell', activations=None,
        ))
        lstm_node = lstm_op.create_node(inputs)
        lstm_node['old_infer'] = lstm_node.infer
        lstm_node.infer = __class__.infer

        # this node consumes one of the resulting LSTMCell outputs,
        # it should be removed before reconnecting the nodes,
        # otherwise it will be reconnected to the new cell output
        graph.remove_node(match['tanh_1'].id)

        for i, output in enumerate(__class__.outputs):
            match[output].replace_node(lstm_node, i)

        # Because of LSTMCell specification, this layer MUST have 2 outputs.
        # => we need to create fake consumers for LSTMCell
        # when this node haven't some outputs.
        for i in [0, 1]:
            if i not in lstm_node.out_nodes():
                fake_output_node = Result(graph, dict(name=lstm_node.name + "/Output_{}".format(i)))
                fake_output_node.create_node(inputs=[lstm_node], edge_attrs={'out': i, 'in': 0})

        lstm_node['tf'] = True
        lstm_node['extra_inputs'] = {name: match[name].id for name in __class__.extra_inputs}
        lstm_node['inputs'] = {name: match[name].id for name in __class__.inputs}

    @staticmethod
    def infer(node: Node):
        assert len(node.in_nodes()) == len(__class__.inputs) + len(__class__.extra_inputs)

        for axis in ['concat_axis', 'split_axis']:
            axis_node = __class__.extra_inputs.index(axis) + len(__class__.inputs)
            assert node.in_node(axis_node).has_valid('value')
            assert node.in_node(axis_node).value == 1

        shift_const = node.in_node(__class__.extra_inputs.index('shift_const') + len(__class__.inputs))
        assert shift_const.has_valid('value')
        shift_const = shift_const.value
        assert shift_const.ndim == 0  # expect scalar value
        node['shift_const'] = shift_const.copy()

        weights_node = node.in_node(__class__.inputs.index('weights'))
        biases_node = node.in_node(__class__.inputs.index('biases'))

        assert weights_node.has_valid('value')
        assert biases_node.has_valid('value')

        # Restore original infer function (to avoid calling previous code twice) and call it
        node.infer = node.old_infer
        node.infer(node)
