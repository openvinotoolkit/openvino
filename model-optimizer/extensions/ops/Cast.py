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

from mo.front.common.partial_infer.elemental import copy_shape_infer
from mo.graph.graph import Node, Graph
from mo.middle.passes.convert_data_type import np_data_type_to_precision, convert_blob, np_data_type_to_destination_type
from mo.ops.op import Op
from mo.utils.utils import refer_to_faq_msg


class Cast(Op):
    op = 'Cast'
    enabled = False

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'op': __class__.op,
            'type': 'Convert',
            'infer': __class__.infer,
            'type_infer': __class__.type_infer,
            'dst_type': None,
            'in_ports_count': 1,
            'out_ports_count': 1,
        }
        super().__init__(graph, mandatory_props, attrs)

    def backend_attrs(self):
        if self.ir_version == 10:
            return [('destination_type', lambda node: np_data_type_to_destination_type(node.dst_type))]
        else:
            return [('precision', lambda node: np_data_type_to_precision(node.dst_type))]

    @staticmethod
    def type_infer(node: Node):
        assert node.has_valid('dst_type'), 'Destination type of "Cast" operation should be extracted earlier'
        node.out_port(0).set_data_type(node.dst_type)

    @staticmethod
    def infer(node: Node):
        assert node.has_valid('dst_type'), 'Destination type of "Cast" operation should be extracted earlier'
        dst_type = node.dst_type
        copy_shape_infer(node)
        if node.has_and_set('stop_value_propagation'):
            return
        if node.in_node(0).has_valid('value'):
            new_blob, finite_match_count, zero_match_count = convert_blob(node.in_node(0).value, dst_type)
            node.out_port(0).data.set_value(new_blob)

            if finite_match_count:
                log.error(
                    ("{} elements of {} were clipped to infinity while converting an input blob for node '{}' to {}. " +
                     refer_to_faq_msg(76)).format(finite_match_count, new_blob.size, node.name, dst_type))
            if zero_match_count:
                log.warning(
                    ("{} elements of {} were clipped to zero while converting an input blob for node '{}' to {}. " +
                     refer_to_faq_msg(77)).format(zero_match_count, new_blob.size, node.name, dst_type))

