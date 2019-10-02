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

from mo.graph.graph import Graph
from mo.middle.pattern_match import apply_pattern


def move_scaleshift_to_preprocess_action(graph, match):
    mean_values = {}
    input_op = match['input_op']
    scale_shift = match['scale_shift']
    weights = np.squeeze(match['weights'].value)
    biases = np.squeeze(match['biases'].value)

    if any([x != 1 for x in weights]):
        return

    # Keep biases (mean values) for current input as graph attr and remove ScaleShift layer
    # Input->data->ScaleShift->scsh_data    =>    Input->scsh_data
    graph.remove_edge(input_op.id, input_op.out_node().id)
    graph.add_edge(input_op.id, scale_shift.out_node().id, out=0)
    graph.remove_edge(scale_shift.id, scale_shift.out_node().id)

    # If bias contains zeros we just remove it
    if all([x == 0 for x in biases]):
        return

    # In pre-process section, mean_values are subtracted
    biases *= -1

    mean_values.update({input_op.name: np.array(biases)})

    # Add graph attribute 'mean_values' that stores mean_values per input if exists
    if graph.graph.get('mean_values', None):
        graph.graph['mean_values'].update(mean_values)
    else:
        graph.graph['mean_values'] = mean_values


def move_scaleshift_to_preprocess(graph: Graph):
    """
    This function finds scaleshift layer after input layer and if it has weights with ones, it deletes scaleshift layer
    and creates graph dict attribute : {'input':np.array(...), 'input2': ... }
    """
    apply_pattern(
        graph,
        nodes=[
            ('weights', dict(kind='data')),
            ('biases', dict(kind='data')),
            ('input_output', dict(kind='data')),
            ('scsh_output', dict(kind='data')),
            ('input_op', dict(kind='op', type='Parameter')),
            ('scale_shift', dict(kind='op', type='ScaleShift')),
        ],
        edges=[
            ('input_op', 'input_output'),
            ('scale_shift', 'scsh_output'),
            ('input_output', 'scale_shift', {'in': 0}),
            ('weights', 'scale_shift', {'in': 1}),
            ('biases', 'scale_shift', {'in': 2}),
        ],
        action=move_scaleshift_to_preprocess_action
    )
