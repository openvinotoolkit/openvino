from typing import Dict

import torch
from torch.nn import Module
from torch.fx import GraphModule
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner

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
        # TODO: this is a naive implementation of cache without proper guard
        self.partitioner_cache: Dict[GraphModule, GraphModule] = {}

        # TODO: this is a naive implementation of cache without proper guard, this will only work for identical inputs
        self.prim_decomp_cache: Dict[GraphModule, GraphModule] = {}

    def fx_serialize(self, graph_module: GraphModule, *args, **kwargs):
        print("Original Graph Module: ", graph_module)
        fx_gm = make_fx(graph_module)(*args)
        #prim_graph = torch.fx.Graph()
        #DecompositionInterpreter(fx_gm, prim_graph, decomposition_table=aten2aten_decomp).run(*args, **kwargs)
        #prim_module = torch.fx.GraphModule(fx_gm, prim_graph)
        return fx_gm #prim_module
   

    def make_partitions(self, graph_module: GraphModule) -> GraphModule:
        # entry function for nvFuser backend
        # logger.debug("Compiling graph_module: ", graph_module.code)
        print("Compiling graph_module: ", graph_module.code)
        # FX graph based partitioning based on nvfuser supported ops
        if graph_module in self.partitioner_cache:
            logger.debug("partitioner_cache hit!")
            print("partitioner_cache hit!")
            fused_graph_module = self.partitioner_cache[graph_module]
        else:
            partitioner = CapabilityBasedPartitioner(
                graph_module, self.supported_ops, allows_single_node_partition=True)
            fused_graph_module = partitioner.partition_and_fuse()
            self.partitioner_cache[graph_module] = fused_graph_module

        return fused_graph_module

    
