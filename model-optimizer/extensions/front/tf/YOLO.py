import networkx as nx

from extensions.front.no_op_eraser import NoOpEraser
from extensions.front.standalone_const_eraser import StandaloneConstEraser
from extensions.ops.regionyolo import RegionYoloOp
from mo.front.tf.replacement import FrontReplacementFromConfigFileGeneral
from mo.graph.graph import Node
from mo.ops.output import Output


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
