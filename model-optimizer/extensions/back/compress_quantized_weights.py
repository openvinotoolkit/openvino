# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Dict

import numpy as np

from extensions.ops.Cast import Cast
from extensions.ops.elementwise import Sub, Div, Mul, Negative, Equal
from extensions.ops.select import Select
from mo.back.replacement import BackReplacementPattern
from mo.graph.graph import Graph, Node
from mo.middle.passes.convert_data_type import data_type_str_to_np, np_data_type_to_destination_type, packed_I4
from mo.middle.pattern_match import apply_pattern
from mo.ops.const import Const


class CompressQuantizeWeights(BackReplacementPattern):
    r"""
    Compress weights transformation goal is to pre-quantize data to minimize runtime calculations with constant data.
    To achieve this goal we perform FakeQuantize decomposition to separate quantization from dequantization in it.

    FakeQuantize:
        -[src_dtype]-> FakeQuantize -[src_dtype]->
    is an operation that could be represented as:
        -[src_dtype]-> Quantize -[quantized_dtype]-> Dequantize -[src_dtype]->

     Quantize and Dequantize operations are not present in OpenVINO supported opsets, but can be easily expressed
     through supported ones. Transformation algorithm doesn't contain all the steps described
     below (some of them are optimized). Steps are presented only to show the idea in details.

    Step 1: FQ decomposition
        -[src_dtype]-> Quantize -[quantized_dtype]-> Dequantize -[src_dtype]->

    Step 2: Representing Quantize and Dequantize through FakeQuantize and Convert operations
        Simplified view:
            -[src_dtype]-> FakeQuantize -[src_dtype]-> Convert -[quantized_dtype]-> Convert -[src_dtype]-> FakeQuantize -[quantized_dtype]->

        Detailed view:
            initial_input_low       initial_input_high               initial_output_low   initial_output_high
                       \                /                                 |              /
                       (in: 1)    (in: 2)                               (in: 3)     (in: 4)
                          V         V                                     V          V
            Constant -> FakeQuantize` --> Convert --> Convert --> initial FakeQuantize -->
                     ^          ^     (quant_dtype)  (src_dtype)           ^         ^
                     |          |                                       (in: 1)    (in: 2)
                (in: 3)    (in: 4)                                         |          |
                   |           \________________          _________________|          |
                   |                            \        /                            |
               new_output_low                 new_output_high                         |
               -(levels // 2)          (levels + new_output_low - 1)                  |
                   |__________________________________________________________________|

    Step 3: All inputs of initial FQ are Constants and we haven't added dynamic dependencies. Means we can const-fold
        sub-graph we already have, but as our goal is to have quantized data, we should mark nodes to be folded.

        -[src_dtype]-> FakeQuantize -[src_dtype]-> Convert -[quantized_dtype]-> Convert -[src_dtype]-> FakeQuantize -[src_dtype]->
        |-------------------------Const Folding-------------------------------|----------------------Stays----------------------------|

        Resulting graph:
            Constant -[quantized_dtype]-> Convert -[src_dtype]-> FakeQuantize -[src_dtype]->

    Step 4: We reduced heavy manipulations with constant data in runtime, but we can go even further.
        At this stage FakeQuantize node is playing dequantization role. It means it only shifts and scales the data.
        No rounding is performed by this FakeQuantize as data was fully quantized earlier.
        Also, runtime calculates this shift (zero point) and scale during low precision transformation.
        It means we can pre-calculate even this information for them by simply decomposing FakeQuantize that plays
        dequantization role to Subtract-Multiply sequence so resulting graph would be:
            Constant -[quantized_dtype]-> Convert -[src_dtype]-> Subtract (zero_point) -> Multiply (scale) -[src_dtype]->

        Where:
            scale = (output_high - output_low) / (input_high - input_low)
                WARNING: division by zero imposes restriction -- input_high can not be equal to input_low
            zero_point = input_low - output_low / scale
            NOTE: if scale == 0 than zero_point is equal to zero too (achieved through Select operation)

    BENEFITS:
        Such constant data packing reduces IR size (.bin file size)
        Also, transformation prepares quantized constant data for Low Precision pipeline.
        With that we can skip same calculations in the runtime and make loading of such sub-graphs to the plugin faster.
    """

    enabled = True
    graph_condition = [lambda graph: not graph.graph['cmd_params'].disable_weights_compression]

    force_clean_up = True

    QUANTIZATION_MAP = {
        # max_levels: (np_dtype, quantization_mode)
        256: (np.int8, "signed"),
        16: (packed_I4, "signed"),
    }

    def pattern(self):
        return dict(
            nodes=[
                ('const', dict(type='Const')),
                ('const_d', dict()),
                ('fake_quantize', dict(type='FakeQuantize', levels=lambda x: x is not None and 2 < x <= 256)),
            ],
            edges=[
                ('const', 'const_d'),
                ('const_d', 'fake_quantize', {'in': 0}),
            ]
        )

    @staticmethod
    def quantize_data(fake_quantize: Node, dst_type: type, quantized_type: type, mode: str):
        graph = fake_quantize.graph
        name = fake_quantize.soft_get('name', fake_quantize.id)
        levels = fake_quantize.levels

        quantize = fake_quantize.copy_node(dict(name=name + '/Copy', stop_value_propagation=False), graph)
        fake_quantize.in_port(0).get_connection().set_destination(quantize.in_port(0))

        # inherit input limits
        fake_quantize.in_port(1).get_connection().set_destination(quantize.in_port(1))
        fake_quantize.in_port(2).get_connection().set_destination(quantize.in_port(2))

        # calculate output limits for quantized weights
        assert mode in ["signed", "unsigned"]
        i_min_value = -(levels // 2) if mode == "signed" else 0

        i_min = np.array([i_min_value], dtype=dst_type)
        i_max = np.array(levels + i_min - 1, dtype=dst_type)

        assert i_max - i_min == levels - 1
        out_low = Const(graph, dict(name=name + '/Copy/out_low', value=i_min)).create_node()
        out_high = Const(graph, dict(name=name + '/Copy/out_high', value=i_max)).create_node()

        out_low.out_port(0).connect(quantize.in_port(3))
        out_high.out_port(0).connect(quantize.in_port(4))
        out_low.out_port(0).connect(fake_quantize.in_port(1))
        out_high.out_port(0).connect(fake_quantize.in_port(2))

        original_const = quantize.in_port(0).get_source().node
        quantized_data_name = original_const.soft_get('name', original_const.id) + '/quantized'
        cast = Cast(graph, dict(name=quantized_data_name, dst_type=quantized_type,
                                stop_value_propagation=False)).create_node()

        quantize.out_port(0).connect(cast.in_port(0))

        cast.out_port(0).connect(fake_quantize.in_port(0))

    @staticmethod
    def dequantize_data(fake_quantize: Node, dst_type: type, quantized_type: type) -> Node:
        graph = fake_quantize.graph
        quantized_data = fake_quantize.in_port(0).get_source().node
        name = fake_quantize.soft_get('name', fake_quantize.id)

        assert quantized_data.soft_get('type') == 'Convert' and quantized_data.dst_type == quantized_type, \
            'Weights aren`t compressed as expected for node {}'.format(fake_quantize.soft_get('name', fake_quantize.id))

        dequantizing_cast = Cast(graph, dict(
            name=quantized_data.name + "/to_{}".format(np_data_type_to_destination_type(dst_type)),
            dst_type=dst_type, stop_value_propagation=True)).create_node()
        fake_quantize.in_port(0).get_connection().set_destination(dequantizing_cast.in_port(0))

        # limits of dequantize
        in_low = fake_quantize.in_port(1).get_source()
        in_high = fake_quantize.in_port(2).get_source()
        out_low = fake_quantize.in_port(3).get_source()
        out_high = fake_quantize.in_port(4).get_source()

        # scale calculation
        output_range = Sub(graph, {'name': name + '/output_range'}).create_node()
        output_range.in_port(0).connect(out_high)
        output_range.in_port(1).connect(out_low)

        input_range = Sub(graph, {'name': name + '/input_range'}).create_node()
        input_range.in_port(0).connect(in_high)
        input_range.in_port(1).connect(in_low)

        scale = Div(graph, {'name': name + '/scale'}).create_node()
        scale.in_port(0).connect(output_range.out_port(0))
        scale.in_port(1).connect(input_range.out_port(0))

        # shift calculation
        descaled_output_low = Div(graph, {'name': name + '/descaled_output_low'}).create_node()
        descaled_output_low.in_port(0).connect(out_low)
        descaled_output_low.in_port(1).connect(scale.out_port(0))

        shift = Sub(graph, {'name': name + '/shift'}).create_node()
        shift.in_port(0).connect(in_low)
        shift.in_port(1).connect(descaled_output_low.out_port(0))

        zero = Const(graph, {'name': name + '/zero', 'value': np.array(0, dtype=dst_type)}).create_node()
        scale_eq_zero = Equal(graph, {'name': name + '/scale_eq_zero'}).create_node()
        scale_eq_zero.in_port(0).connect(scale.out_port(0))
        scale_eq_zero.in_port(1).connect(zero.out_port(0))

        if_scale_is_zero = Select(graph, {'name': name + '/zero_point'}).create_node()
        if_scale_is_zero.in_port(0).connect(scale_eq_zero.out_port(0))
        if_scale_is_zero.in_port(1).connect(zero.out_port(0))
        if_scale_is_zero.in_port(2).connect(shift.out_port(0))

        # DeQuantize(x) == Mul(Sub(x, zero_point), scale)
        sub_zp = Sub(graph, {'name': name + '/minus_zp'}).create_node()
        sub_zp.in_port(0).connect(dequantizing_cast.out_port(0))
        sub_zp.in_port(1).connect(if_scale_is_zero.out_port(0))

        mul_scale = Mul(graph, {'name': name + '/mulpiply_by_scale'}).create_node()
        mul_scale.in_port(0).connect(sub_zp.out_port(0))
        mul_scale.in_port(1).connect(scale.out_port(0))

        fake_quantize.out_port(0).get_connection().set_source(mul_scale.out_port(0))

        graph.remove_nodes_from([fake_quantize.id, fake_quantize.out_node(0)])

    def replace_pattern(self, graph: Graph, match: Dict[str, Node]):
        fake_quantize = match['fake_quantize']

        dst_type = match['const'].value.dtype
        if np.issubdtype(dst_type, np.floating):
            dst_type = data_type_str_to_np(graph.graph['cmd_params'].data_type)

        quantized_type, mode = None, None
        for quantization_levels in sorted(self.QUANTIZATION_MAP):
            if quantization_levels >= fake_quantize.levels:
                quantized_type, mode = self.QUANTIZATION_MAP[quantization_levels]
                break

        self.quantize_data(fake_quantize, dst_type, quantized_type, mode)
        self.dequantize_data(fake_quantize, dst_type, quantized_type)


class ZeroPointOptimizer(BackReplacementPattern):
    r"""
    Step 1: Having zero_point == 0 is really beneficial for performance, so we try to fuse Subtract up to the Constant.
        It is not always possible because of the quantized_dtype possible range of values.

    Step 2: From the nature of Subtract operation it may be optimized out if zero_point == 0
    """
    enabled = True
    force_clean_up = True

    def run_after(self):
        return [CompressQuantizeWeights]

    def pattern(self):
        return dict(
            nodes=[
                ('const', dict(type='Const')),
                ('const_d', dict()),
                ('convert', dict(type='Convert')),
                ('convert_d', dict()),
                ('const_zp', dict(type='Const')),
                ('const_zp_d', dict()),
                ('sub', dict(type='Subtract')),
            ],
            edges=[
                ('const', 'const_d'),
                ('const_d', 'convert'),
                ('convert', 'convert_d'),
                ('convert_d', 'sub', {'in': 0}),
                ('const_zp', 'const_zp_d'),
                ('const_zp_d', 'sub', {'in': 1}),
            ]
        )

    def replace_pattern(self, graph: Graph, match: Dict[str, Node]):
        zero_point = match['const_zp'].out_port(0).data.get_value()
        assert zero_point is not None
        convert = match['convert']
        sub = match['sub']
        if np.allclose(zero_point, 0):
            sub.out_port(0).get_connection().set_source(convert.out_port(0))
            return

        weights = match['const'].out_port(0).data.get_value()
        if weights is None or weights.dtype != np.int8:
            return
        dst_type = convert.dst_type

        int8_zero_point = np.round(zero_point).astype(np.int8)
        adj_zero_point = (zero_point - int8_zero_point).astype(dst_type)

        original = weights.astype(dst_type) - zero_point
        transformed = (weights - int8_zero_point).astype(np.int8) - adj_zero_point

        if not np.allclose(original, transformed) or not np.allclose(adj_zero_point, 0, atol=1.e-04):
            return

        match['const_d']['value'] = (weights - int8_zero_point).astype(np.int8)
        sub.out_port(0).get_connection().set_source(convert.out_port(0))
