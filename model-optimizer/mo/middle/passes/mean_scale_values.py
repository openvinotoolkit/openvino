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

import numpy as np

from mo.graph.graph import Graph
from mo.middle.pattern_match import apply_pattern


def move_mean_values_to_preprocess_action(graph, match):
    mean_values = {}
    input_op = match['input_op']
    add_op = match['add']
    mul_op = match['mul']

    if match['const_mul_output'].has('value'):
        weights = match['const_mul_output'].value
        if any([x != 1 for x in weights]):
            return

    # # Keep biases (mean values) for current input as graph attr and remove mean values layers
    # # Input->data->Mul->Add->scsh_data    =>    Input->scsh_data
    add_op.out_port(0).get_connection().set_source(input_op.out_port(0))
    mul_op.in_port(0).disconnect()

    if match['const_add_output'].has('value'):
        biases = match['const_add_output'].value

        if graph.graph['cmd_params'].reverse_input_channels:
            biases = np.flip(biases)

        # If bias contains zeros we just remove it
        if all([x == 0 for x in biases]):
            return

        biases *= -1
        # In pre-process section, mean_values are subtracted
        mean_values.update({input_op.name: np.array(biases)})

        # Add graph attribute 'mean_values' that stores mean_values per input if exists
        if graph.graph.get('mean_values', None):
            graph.graph['mean_values'].update(mean_values)
        else:
            graph.graph['mean_values'] = mean_values


def move_mean_values_to_preprocess(graph: Graph):
    """
    This function finds mean values layers after input layer and if it has weights with ones, it deletes mean values layers
    and creates graph dict attribute : {'input':np.array(...), 'input2': ... }
    """
    apply_pattern(
        graph,
        nodes = [
            ('input_op', dict(kind='op', op='Parameter')),
            ('input_output', dict(kind='data')),

            ('const_mul', dict(kind='op', op='Const')),
            ('const_mul_output', dict(kind='data')),
            ('mul', dict(kind='op', op='Mul')),
            ('mul_output', dict(kind='data')),

            ('const_add', dict(kind='op', op='Const')),
            ('const_add_output', dict(kind='data')),
            ('add', dict(kind='op', op='Add')),
            ('add_output', dict(kind='data')),
        ],
        edges = [
            ('input_op', 'input_output'),
            ('input_output', 'mul', {'in': 0}),
            ('const_mul', 'const_mul_output'),
            ('const_mul_output', 'mul', {'in': 1}),
            ('mul', 'mul_output'),

            ('mul_output', 'add', {'in': 0}),
            ('const_add', 'const_add_output'),
            ('const_add_output', 'add', {'in': 1}),
            ('add', 'add_output'),
        ],
        action=move_mean_values_to_preprocess_action
    )
