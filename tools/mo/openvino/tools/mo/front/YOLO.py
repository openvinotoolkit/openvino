# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.no_op_eraser import NoOpEraser
from openvino.tools.mo.ops.regionyolo import RegionYoloOp
from openvino.tools.mo.front.tf.replacement import FrontReplacementFromConfigFileGeneral
from openvino.tools.mo.graph.graph import Node, Graph
from openvino.tools.mo.ops.result import Result
from openvino.tools.mo.utils.error import Error


class YoloRegionAddon(FrontReplacementFromConfigFileGeneral):
    """
    Replaces all Result nodes in graph with YoloRegion->Result nodes chain.
    YoloRegion node attributes are taken from configuration file
    """
    replacement_id = 'TFYOLO'

    def run_after(self):
        return [NoOpEraser]

    def transform_graph(self, graph: Graph, replacement_descriptions):
        op_outputs = [n for n, d in graph.nodes(data=True) if 'op' in d and d['op'] == 'Result']
        for op_output in op_outputs:
            last_node = Node(graph, op_output).in_node(0)
            op_params = dict(name=last_node.id + '/YoloRegion', axis=1, end_axis=-1, nchw_layout=True)
            op_params.update(replacement_descriptions)
            region_layer = RegionYoloOp(graph, op_params)
            region_layer_node = region_layer.create_node([last_node])
            # here we remove 'axis' from 'dim_attrs' to avoid permutation from axis = 1 to axis = 2
            region_layer_node.dim_attrs.remove('axis')
            Result(graph).create_node([region_layer_node])
            graph.remove_node(op_output)


class YoloV3RegionAddon(FrontReplacementFromConfigFileGeneral):
    """
    Replaces all Result nodes in graph with YoloRegion->Result nodes chain.
    YoloRegion node attributes are taken from configuration file
    """
    replacement_id = 'TFYOLOV3'

    def transform_graph(self, graph: Graph, replacement_descriptions):
        graph.remove_nodes_from(graph.get_nodes_with_attributes(op='Result'))
        for i, input_node_name in enumerate(replacement_descriptions['entry_points']):
            if input_node_name not in graph.nodes():
                raise Error('TensorFlow YOLO V3 conversion mechanism was enabled. '
                            'Entry points "{}" were provided in the configuration file. '
                            'Entry points are nodes that feed YOLO Region layers. '
                            'Node with name {} doesn\'t exist in the graph. '
                            'Refer to documentation about converting YOLO models for more information.'.format(
                    ', '.join(replacement_descriptions['entry_points']), input_node_name))
            last_node = Node(graph, input_node_name).in_node(0)
            op_params = dict(name=last_node.id + '/YoloRegion', axis=1, end_axis=-1, do_softmax=0, nchw_layout=True)
            op_params.update(replacement_descriptions)
            if 'masks' in op_params:
                op_params['mask'] = op_params['masks'][i]
                del op_params['masks']
            region_layer_node = RegionYoloOp(graph, op_params).create_node([last_node])
            # TODO: do we need change axis for further permutation
            region_layer_node.dim_attrs.remove('axis')
            Result(graph, {'name': region_layer_node.id + '/Result'}).create_node([region_layer_node])
