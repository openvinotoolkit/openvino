# -*- coding: utf-8 -*-
# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# mypy: ignore-errors

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
from openvino.frontend.pytorch.torchdynamo.backend_utils import _is_testing

import typing as t
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class Partitioner:
    def __init__(self, options):
        self.supported_ops = OperatorSupport(options)

    def fx_serialize(self, graph_module: GraphModule, *args, **kwargs):
        fx_gm = make_fx(graph_module)(*args)
        return fx_gm

    def add_get_attr_inputs(self, partitions: t.List[Partition]):
        # TODO: Find a more efficient way to include input
        # "get_attr" nodes to the partitions.
        getattr_to_merge: Dict[Node, Node] = {}
        for partition in partitions:
            for pnode in partition.nodes:
                for pnode_input in pnode.all_input_nodes:
                    if pnode_input.op in ["get_attr"] and pnode_input.op not in getattr_to_merge:
                        getattr_to_merge[pnode_input] = partition
        for getattr_node, getattr_part in getattr_to_merge.items():
            getattr_part.add_node(getattr_node)

    def check_fully_supported(self, graph_module: GraphModule) -> bool:
        num_fused = 0
        for node in graph_module.graph.nodes:
            if node.op == "call_module" and "fused_" in node.name:
                num_fused += 1
            elif node.op != "placeholder" and node.op != "output":
                return False
        if num_fused == 1:
            return True
        return False

    def make_partitions(self, graph_module: GraphModule, options) -> GraphModule:
        allow_single_node_partition = _is_testing(options)
        partitioner = CapabilityBasedPartitioner(
            graph_module, self.supported_ops, allows_single_node_partition=allow_single_node_partition)
        partitions = partitioner.propose_partitions()
        self.add_get_attr_inputs(partitions)
        fused_graph_module = partitioner.fuse_partitions(partitions)

        return fused_graph_module
