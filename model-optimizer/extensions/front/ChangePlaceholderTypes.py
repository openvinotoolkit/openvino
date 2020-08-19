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

from mo.front.common.replacement import FrontReplacementPattern
from mo.graph.graph import Graph, Node


class ChangePlaceholderTypes(FrontReplacementPattern):
    enabled = True

    @staticmethod
    def is_node_casts_to_float_or_shapeof(node: Node):
        return (node.soft_get('type') == 'Convert' and node.soft_get('dst_type') == np.float32) or \
                node.soft_get('type') == 'ShapeOf'

    def find_and_replace_pattern(self, graph: Graph):
        for op in graph.get_op_nodes(type='Parameter'):
            consumer_nodes = [p.node for p in op.out_port(0).get_destinations()]
            if all([ChangePlaceholderTypes.is_node_casts_to_float_or_shapeof(consumer) for consumer in consumer_nodes]):
                log.debug('Convert data type of Parameter "{}" to float32'.format(op.soft_get('name', op.id)))
                op.data_type = np.float32
                for convert_node in consumer_nodes:
                    if convert_node.soft_get('type') == 'Convert':
                        log.debug('Removing "Convert" node "{}"'.format(convert_node.soft_get('name', convert_node.id)))

                        # disconnect consumer ports of Convert operations. Then connect them with an output of Parameter
                        convert_destinations = convert_node.out_port(0).get_destinations()
                        for dst_port in convert_destinations:
                            dst_port.disconnect()
                        for dst_port in convert_destinations:
                            op.out_port(0).connect(dst_port)

                        graph.remove_node(convert_node.id)

            if op.soft_get('data_type') == np.uint8:
                op.data_type = np.float32
                log.debug('Convert data type of Parameter "{}" to float'.format(op.soft_get('name', op.id)))
