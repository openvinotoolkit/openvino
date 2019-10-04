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
from typing import Dict

from extensions.ops.elementwise import Mul, Add
from mo.graph.graph import Graph, Node
from mo.back.replacement import BackReplacementPattern
from mo.ops.const import Const


class WeightsQuantize(BackReplacementPattern):
    """
    Allows to store constant weights as uint8(or uint4 if it's possible) data type instead fp32.
    The structure of pattern without Data nodes.
    Detects pattern:

    -------------------------------------------------------------
    | fp32_weights ----> Initial_FakeQuantize ----> Convolution |
    -------------------------------------------------------------

    and transforms it to:

    ------------------------------------------------------------------------------------------------
    |                                   Scale     Shift                                            |
    |                                     |         |                                              |
    |                                     |         |                                              |
    |                                     V         V                                              |
    | fp32_weights ----> Quantize ----> Mul ----> Add ----> Initial_FakeQuantize ----> Convolution |
    ------------------------------------------------------------------------------------------------

    input_low/input_high for Quantize op has the same value as output_low/output_high for Initial_FakeQuantize.

    After value propagation:

    ----------------------------------------------------------------------------------------------------------------
    |                                                  Scale     Shift                                             |
    |                                                    |         |                                               |
    |                                                    |         |                                               |
    |                                                    V         V                                               |
    | fp32_weights (with force_precision to int8) ----> Mul ----> Add ----> Initial_FakeQuantize ----> Convolution |
    ----------------------------------------------------------------------------------------------------------------
    """

    enabled = True
    force_clean_up = True

    def pattern(self):
        return dict(
            nodes=[
                ('weights_const', dict(type='Const')),
                ('weights_d', dict(kind='data')),
                ('quantize', dict(type='FakeQuantize', keep_in_IR=True, levels=lambda x: x is not None and
                                                                                         2 < x <= 256)),
                ('quantize_d', dict(kind='data')),
                ('convolution', dict(type='Convolution'))],
            edges=[
                ('weights_const', 'weights_d'),
                ('weights_d', 'quantize', {'in': 0}),
                ('quantize', 'quantize_d'),
                ('quantize_d', 'convolution', {'in': 1})
            ]
        )

    def replace_pattern(self, graph: Graph, match: Dict[str, Node]):

        quantize = match['quantize']
        i_min = 0

        # 1. Set input_low and input_high values to int8 quantize operation (min/max fp32 values)
        new_quantize_node = quantize.copy_node(dict(name=quantize.name + '/Copy', stop_value_propagation=False), graph)

        quantize.in_port(1).get_source().connect(new_quantize_node.in_port(1))
        quantize.in_port(2).get_source().connect(new_quantize_node.in_port(2))

        # 2. Set the interval of integer values to int8 quantize operation
        new_out_low_node = Const(graph, dict(name=quantize.name + '/Copy/out_low', value=i_min)).create_node()
        new_out_high_node = Const(graph, dict(name=quantize.name + '/Copy/out_high',
                                              value=quantize.levels - 1)).create_node()

        new_out_low_node.out_port(0).connect(new_quantize_node.in_port(3))
        new_out_high_node.out_port(0).connect(new_quantize_node.in_port(4))

        # 3. Create Mul and Add operations with relevant scale and shift
        # fp32 values might be received from uint8 values by formula:
        # fp32_value = uint8_value * scale + shift
        f_min = quantize.in_port(3).get_source().data.get_value()
        f_max = quantize.in_port(4).get_source().data.get_value()
        assert f_min is not None and f_max is not None, 'Value of FakeQuantize range input is dynamic, can not proceed'
        scale = (f_max - f_min) / (quantize.levels - 1 - i_min)
        shift = f_min - scale * i_min
        new_scale_node = Const(graph, dict(name="Scale", value=scale)).create_node()
        new_shift_node = Const(graph, dict(name="Shift", value=shift)).create_node()
        new_mul_node = Mul(graph, dict(name=quantize.name + "/Scale", stop_value_propagation=True)).create_node()
        new_add_node = Add(graph, dict(name=quantize.name + "/Shift", stop_value_propagation=True)).create_node()

        new_scale_node.out_port(0).connect(new_mul_node.in_port(0))
        new_shift_node.out_port(0).connect(new_add_node.in_port(0))

        # 4. Set connection: int8 Quantization -> Mul -> Add
        new_mul_node.out_port(0).connect(new_add_node.in_port(1))
        new_quantize_node.out_port(0).connect(new_mul_node.in_port(1))

        # 5. Connect new subgraph to the network
        quantize.in_port(0).get_connection().set_destination(new_quantize_node.in_port(0))
        new_add_node.out_port(0).connect(quantize.in_port(0))

        if quantize.levels <= 16:
            new_mul_node['force_precision_in_ports'] = {1: 'uint4'}
        else:
            new_mul_node['force_precision_in_ports'] = {1: 'uint8'}
