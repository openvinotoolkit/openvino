"""
 Copyright (c) 2020 Intel Corporation

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

from typing import Dict

import numpy as np

from extensions.ops.Cast import Cast
from mo.back.replacement import BackReplacementPattern
from mo.graph.graph import Graph, Node
from mo.middle.passes.convert_data_type import data_type_str_to_np
from mo.ops.const import Const


class CompressQuantizeWeights(BackReplacementPattern):
    """
    Allows to store constant weights as uint8 data type instead fp32.
    The structure of pattern without Data nodes.
    Detects pattern:
    ------------------------------------------
    | fp32_weights ---> Initial_FakeQuantize |
    ------------------------------------------

    But actually it looks like:

    ---------------------------------------------------------------------------
    |                                                                         |
    |                                                                         |
    |         initial_input_low               initial_input_high              |
    |                       \                 /                               |
    |                        \               /                                |
    |                      (in: 1)        (in: 2)                             |
    |                          V           V                                  |
    | fp32_weights ----> Initial_FakeQuantize                                 |
    |                     ^                 ^                                 |
    |                 (in: 3)              (in: 4)                            |
    |                  /                      \                               |
    |                /                         \                              |
    |    initial_output_low                initial_output_high                |
    |                                                                         |
    |                                                                         |
    ---------------------------------------------------------------------------

    And transforms it to:

    -------------------------------------------------------------------------------------------------------------
    |                                                                                                           |
    |       initial_input_low       initial_input_high                initial_output_low   initial_output_high  |
    |                  \                /                                  |              /                     |
    |                   \              /                                   |            /                       |
    |                  (in: 1)    (in: 2)                                (in: 3)     (in: 4)                    |
    |                     V         V                                      V          V                         |
    | fp32_weights ----> FakeQuantize ----> Convert (to fp32) ----> Initial_FakeQuantize                        |
    |               (with int8 output type)                            ^            ^                           |
    |                   ^            ^                             (in: 1)        (in: 2)                       |
    |                (in: 3)        (in: 4)                          |              |                           |
    |                   |              \           ------------------               |                           |
    |                   |               \        /                                  |                           |
    |               output_low         output_high                                  |                           |
    |                  (0)            (levels - 1)                                  |                           |
    |                   |                                                           |                           |
    |                   |                                                           |                           |
    |                   -------------------------------------------------------------                           |
    |                                                                                                           |
    |                                                                                                           |
    |                                                                                                           |
    -------------------------------------------------------------------------------------------------------------

    Initial_FakeQuantize will restore original fp32 values during inference.

    After value propagation the sub-graph will look like:

    uint8_weights ---> Convert (to fp32) ---> Initial_FakeQuantize

    """

    enabled = True
    graph_condition = [lambda graph: not graph.graph['cmd_params'].disable_weights_compression]

    force_clean_up = True

    def pattern(self):
        return dict(
            nodes=[
                ('weights_const', dict(type='Const')),
                ('weights_d', dict(kind='data')),
                ('quantize', dict(type='FakeQuantize', levels=lambda x: x is not None and 2 < x <= 256)),
                ('quantize_d', dict(kind='data')),
                ('convolution', dict())
            ],
            edges=[
                ('weights_const', 'weights_d'),
                ('weights_d', 'quantize', {'in': 0}),
                ('quantize', 'quantize_d'),
                ('quantize_d', 'convolution', {'in': 1})
            ]
        )

    def replace_pattern(self, graph: Graph, match: Dict[str, Node]):
        initial_fake_quantize = match['quantize']

        new_fake_quantize = initial_fake_quantize.copy_node(dict(name=initial_fake_quantize.name + '/Copy',
                                                                 stop_value_propagation=False), graph)

        initial_fake_quantize.in_port(1).get_connection().set_destination(new_fake_quantize.in_port(1))
        initial_fake_quantize.in_port(2).get_connection().set_destination(new_fake_quantize.in_port(2))

        dst_type = data_type_str_to_np(graph.graph['cmd_params'].data_type)

        i_min = np.array([0.], dtype=dst_type)
        i_max = np.array([initial_fake_quantize.levels - 1.], dtype=dst_type)

        new_out_low_node = Const(graph, dict(name=initial_fake_quantize.name + '/Copy/out_low',
                                             value=i_min)).create_node()
        new_out_high_node = Const(graph, dict(name=initial_fake_quantize.name + '/Copy/out_high',
                                              value=i_max)).create_node()

        new_out_low_node.out_port(0).connect(new_fake_quantize.in_port(3))
        new_out_high_node.out_port(0).connect(new_fake_quantize.in_port(4))
        new_out_low_node.out_port(0).connect(initial_fake_quantize.in_port(1))
        new_out_high_node.out_port(0).connect(initial_fake_quantize.in_port(2))

        cast_node = Cast(graph, dict(name=initial_fake_quantize.name + "/Convert_to_float", dst_type=dst_type,
                                     stop_value_propagation=True)).create_node()
        new_fake_quantize.out_port(0).connect(cast_node.in_port(0))
        initial_fake_quantize.in_port(0).get_connection().set_destination(new_fake_quantize.in_port(0))
        cast_node.out_port(0).connect(initial_fake_quantize.in_port(0))

        cast_node['force_precision_in_ports'] = {0: 'uint8'}
