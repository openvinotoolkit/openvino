"""
 Copyright (c) 2018 Intel Corporation

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
import networkx as nx

from extensions.front.no_op_eraser import NoOpEraser
from extensions.front.standalone_const_eraser import StandaloneConstEraser
from extensions.ops.regionyolo import RegionYoloOp
from mo.front.tf.replacement import FrontReplacementFromConfigFileGeneral
from mo.graph.graph import Node
from mo.middle.passes.eliminate import get_nodes_with_attributes
from mo.ops.output import Output
from mo.utils.error import Error


class YoloRegionAddon(FrontReplacementFromConfigFileGeneral):
    """
    Replaces all OpOutput nodes in graph with YoloRegion->OpOutput nodes chain.
    YoloRegion node attributes are taken from configuration file
    """
    replacement_id = 'TFYOLO'

    def run_after(self):
        return [NoOpEraser, StandaloneConstEraser]

    def transform_graph(self, graph: nx.MultiDiGraph, replacement_descriptions):
        op_outputs = [n for n, d in graph.nodes(data=True) if 'op' in d and d['op'] == 'OpOutput']
        for op_output in op_outputs:
            last_node = Node(graph, op_output).in_node(0)
            op_params = dict(name=last_node.id + '/YoloRegion', axis=1, end_axis=-1)
            op_params.update(replacement_descriptions)
            region_layer = RegionYoloOp(graph, op_params)
            region_layer_node = region_layer.create_node([last_node])
            # here we remove 'axis' from 'dim_attrs' to avoid permutation from axis = 1 to axis = 2
            region_layer_node.dim_attrs.remove('axis')
            Output(graph).create_node([region_layer_node])


class YoloV3RegionAddon(FrontReplacementFromConfigFileGeneral):
    """
    Replaces all OpOutput nodes in graph with YoloRegion->OpOutput nodes chain.
    YoloRegion node attributes are taken from configuration file
    """
    replacement_id = 'TFYOLOV3'

    def transform_graph(self, graph: nx.MultiDiGraph, replacement_descriptions):
        graph.remove_nodes_from(get_nodes_with_attributes(graph, is_output=True))
        for input_node_name in replacement_descriptions['entry_points']:
            if input_node_name not in graph.nodes():
                raise Error('TensorFlow YOLO V3 conversion mechanism was enabled. '
                            'Entry points "{}" were provided in the configuration file. '
                            'Entry points are nodes that feed YOLO Region layers. '
                            'Node with name {} doesn\'t exist in the graph. '
                            'Refer to documentation about converting YOLO models for more information.'.format(
                    ', '.join(replacement_descriptions['entry_points']), input_node_name))
            last_node = Node(graph, input_node_name).in_node(0)
            op_params = dict(name=last_node.id + '/YoloRegion', axis=1, end_axis=-1, do_softmax=0, is_output=True)
            op_params.update(replacement_descriptions)
            region_layer_node = RegionYoloOp(graph, op_params).create_node([last_node])
            # TODO: do we need change axis for further permutation
            region_layer_node.dim_attrs.remove('axis')
            Output(graph, {'name': region_layer_node.id + '/OpOutput'}).create_node([region_layer_node])
