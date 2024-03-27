# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import reverse_bypass_infer
from openvino.tools.mo.graph.graph import Node, Graph
from openvino.tools.mo.middle.passes.convert_data_type import np_data_type_to_precision, convert_blob, \
    np_data_type_to_destination_type, packed_I4, packed_U4
from openvino.tools.mo.ops.op import Op
from openvino.tools.mo.utils.utils import refer_to_faq_msg


class Cast(Op):
    op = 'Cast'
    enabled = False

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'op': self.op,
            'type': 'Convert',
            'version': 'opset1',
            'infer': self.infer,
            'reverse_infer': lambda node: reverse_bypass_infer(node, in_ports=[0]),
            'type_infer': self.type_infer,
            'dst_type': None,
            'in_ports_count': 1,
            'out_ports_count': 1,
        }
        super().__init__(graph, mandatory_props, attrs)

    def backend_attrs(self):
        return [('destination_type', lambda node: np_data_type_to_destination_type(node.dst_type))]

    @staticmethod
    def type_infer(node: Node):
        assert node.has_valid(
            'dst_type'), 'Destination type of "Cast" operation should be extracted earlier'
        node.out_port(0).set_data_type(node.dst_type)

    @staticmethod
    def helper_value_propagation(node_name, value, dst_type):
        new_blob, finite_match_count, zero_match_count = convert_blob(
            value, dst_type)

        if finite_match_count:
            log.error("{} elements of {} were clipped to infinity while converting an input blob for node '{}' to {}."
                      " ".format(finite_match_count, new_blob.size, node_name, dst_type) + refer_to_faq_msg(76))
        if zero_match_count:
            log.warning("{} elements of {} were clipped to zero while converting an input blob for node '{}' to {}."
                        " ".format(zero_match_count, new_blob.size, node_name, dst_type) + refer_to_faq_msg(77))
        return new_blob

    @staticmethod
    def custom_type_casting_and_packing(node: Node, value, dst_type):
        """
        Custom types are not supported by numpy but we still need to write it to the .bin file in a compact way.
        To do so we prepare bit representation of int4/uint4 values and store them in a numpy friendly data type.
        We pack int4/uint4 values into uint8 type (two int4/uint4 numbers fit in uint8).
        If the number of elements in the blob is odd we pad them with zero value to be able to fit the bit sequence
        into the uint8 array.
        Example: we need to represent 5 elements of int4 dtype
            we would pad them to 6 element with the last element as zero and we would pack them into 3 uint8 values
        """
        assert dst_type in [packed_U4, packed_I4]
        # TODO: Remove this comment when it's clear that we can fix it easily
        # raise Exception("Packing of u4/i4 data is no longer supported in mo because it is now incompatible with the new "
        #                 "order of the halfs of a byte that was introduced in OpenVINO runtime recently. Use ovc "
        #                 "command line tool or openvino.convert_model python function instead.")

        minimum_regular_dtype = np.uint8 if dst_type == packed_U4 else np.int8
        # initial casing from the source type to the numpy-friendly type which could absorb all the values of dst_type
        casted_to_regular_type = Cast.helper_value_propagation(
            node.soft_get('name', node.id), value, minimum_regular_dtype)

        # packing the values
        data_shape = node.out_port(0).data.get_shape()
        assert data_shape is not None
        data_size = np.prod(data_shape)

        num_bits = 4
        assert num_bits < 8 and 8 % num_bits == 0, "Packing algorithm for the data types stored in 1, 2 or 4 bits"
        num_values_fitting_into_uint8 = 8 // num_bits
        pad = (-data_size) % num_values_fitting_into_uint8

        flattened = casted_to_regular_type.flatten()
        padded = np.concatenate((flattened, np.zeros([pad], dtype=minimum_regular_dtype)))
        assert np.prod(padded.shape) % num_values_fitting_into_uint8 == 0

        bit_order_little = (padded[:, None] & (
            1 << np.arange(num_bits)) > 0).astype(np.uint8)
        bit_order_big_flattened = bit_order_little.flatten()
        # u1 still has reversed bit order:
        packed = np.packbits(bit_order_big_flattened,
                             bitorder='little' if num_bits > 1 else 'big')

        node.out_node(0)['force_shape'] = data_shape.copy()
        node.out_node(0)['force_type'] = np_data_type_to_precision(dst_type)
        node.out_port(0).data.set_value(packed)

    @staticmethod
    def infer(node: Node):
        node_name = node.soft_get('name', node.id)
        dst_type = node.soft_get('dst_type', None)

        assert dst_type is not None, \
            'Destination type of "Cast" operation should be extracted earlier, but it`s not for node: ' + node_name

        input_shape = node.in_port(0).data.get_shape()
        assert input_shape is not None
        node.out_port(0).data.set_shape(input_shape)

        value = node.in_port(0).data.get_value()
        if value is None or node.has_and_set('stop_value_propagation'):
            return

        if dst_type in [packed_U4, packed_I4]:  # custom types conversion
            Cast.custom_type_casting_and_packing(node, value, dst_type)
        else:
            node.out_port(0).data.set_value(
                Cast.helper_value_propagation(node_name, value, dst_type))
