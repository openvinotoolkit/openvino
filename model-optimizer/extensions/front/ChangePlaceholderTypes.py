# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

import numpy as np

from mo.front.common.replacement import FrontReplacementPattern
from mo.graph.graph import Graph, Node


class ChangePlaceholderTypes(FrontReplacementPattern):
    enabled = True
    run_not_recursively = True

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
            if op.soft_get('data_type') == np.int64:
                op.data_type = np.int32
                log.error('Convert data type of Parameter "{}" to int32'.format(op.soft_get('name', op.id)),
                          extra={'is_warning': True})

            if op.soft_get('data_type') == np.uint8:
                op.data_type = np.float32
                log.debug('Convert data type of Parameter "{}" to float'.format(op.soft_get('name', op.id)))
