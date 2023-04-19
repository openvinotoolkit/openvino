from typing import Dict

import torch
from torch.nn import Module
from torch.fx import GraphModule, Node
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner, Partition

from torch.fx.experimental.proxy_tensor import DecompositionInterpreter
from torch._decomp import decomposition_table
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten
from openvino.frontend.pytorch.torchdynamo.op_support import OperatorSupport

import typing as t
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class Partitioner:
    def __init__(self):
        self.supported_ops = OperatorSupport()

    def fx_serialize(self, graph_module: GraphModule, *args, **kwargs):
        print("Original Graph Module: ", graph_module)
        fx_gm = make_fx(graph_module)(*args)
        #prim_graph = torch.fx.Graph()
        #DecompositionInterpreter(fx_gm, prim_graph, decomposition_table=aten2aten_decomp).run(*args, **kwargs)
        #prim_module = torch.fx.GraphModule(fx_gm, prim_graph)
        return fx_gm #prim_module

    def add_get_attr_inputs(self, partitions: t.List[Partition]):   
        #TODO: Find a more efficient way to include input
        #"get_attr" nodes to the partitions.
        getattr_to_merge : Dict[Node, Node] = {}
        for partition in partitions:
            for pnode in partition.nodes:
                for pnode_input in pnode.all_input_nodes:
                    if pnode_input.op in ['get_attr']:
                        if pnode_input.op not in getattr_to_merge:
                            getattr_to_merge[pnode_input] = partition
        for getattr_node, getattr_part in getattr_to_merge.items():
            getattr_part.add_node(getattr_node)

    def make_partitions(self, graph_module: GraphModule) -> GraphModule:
        # entry function for nvFuser backend
        # logger.debug("Compiling graph_module: ", graph_module.code)
        print("Compiling graph_module: ", graph_module.code)
        # FX graph based partitioning based on nvfuser supported ops
        partitioner = CapabilityBasedPartitioner(
            graph_module, self.supported_ops, allows_single_node_partition=False)
        partitions = partitioner.propose_partitions()
        self.add_get_attr_inputs(partitions)
        fused_graph_module = partitioner.fuse_partitions(partitions)

        return fused_graph_module

    
