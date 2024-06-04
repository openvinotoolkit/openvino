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

    def capture_gptq_patterns(self, graph_module: GraphModule) -> bool:
        for node in graph_module.graph.nodes:
            if str(node.op) == "call_function" and str(node.target) == "aten.bitwise_and.Scalar":
                bitwise_and_in_nodes = node.all_input_nodes
                if len(bitwise_and_in_nodes) != 1:
                    continue
                to_copy_node = bitwise_and_in_nodes[0]
                if str(to_copy_node.op) != "call_function" or str(to_copy_node.target) != "aten._to_copy.default":
                    continue
                to_copy_in_nodes = to_copy_node.all_input_nodes
                if len(to_copy_in_nodes) != 1:
                    continue
                bitwise_right_shift_node = to_copy_in_nodes[0]
                if str(bitwise_right_shift_node.op) != "call_function" or str(bitwise_right_shift_node.target) != "aten.bitwise_right_shift.Tensor":
                    continue
                bitwise_right_shift_in_nodes = bitwise_right_shift_node.all_input_nodes
                if len(bitwise_right_shift_in_nodes) != 2:
                    continue
                expand_node = bitwise_right_shift_in_nodes[0]
                if str(expand_node.op) != "call_function" or str(expand_node.target) != "aten.expand.default":
                    continue
                expand_in_nodes = expand_node.all_input_nodes
                if len(expand_in_nodes) != 1:
                    continue
                unsqueeze_0_node = expand_in_nodes[0]
                if str(unsqueeze_0_node.op) != "call_function" or str(unsqueeze_0_node.target) != "aten.unsqueeze.default":
                    continue
                unsqueeze_0_in_nodes = unsqueeze_0_node.all_input_nodes
                if len(unsqueeze_0_in_nodes) != 1:
                    continue
                const_0_node = unsqueeze_0_in_nodes[0]
                if str(const_0_node.op) != "get_attr":
                    continue
                unsqueeze_1_node = bitwise_right_shift_in_nodes[1]
                if str(unsqueeze_1_node.op) != "call_function" or str(unsqueeze_1_node.target) != "aten.unsqueeze.default":
                    continue
                unsqueeze_1_in_nodes = unsqueeze_1_node.all_input_nodes
                if len(unsqueeze_1_in_nodes) != 1:
                    continue
                const_1_node = unsqueeze_1_in_nodes[0]
                if str(const_1_node.op) != "get_attr":
                    continue

                self.supported_ops.enable_by_name(node)
                self.supported_ops.enable_by_name(to_copy_node)
                self.supported_ops.enable_by_name(bitwise_right_shift_node)
                self.supported_ops.enable_by_name(expand_node)
                self.supported_ops.enable_by_name(unsqueeze_0_node)
                self.supported_ops.enable_by_name(unsqueeze_1_node)

    def make_partitions(self, graph_module: GraphModule, options) -> GraphModule:
        allow_single_node_partition = _is_testing(options)
        self.capture_gptq_patterns(graph_module)
        partitioner = CapabilityBasedPartitioner(
            graph_module, self.supported_ops, allows_single_node_partition=allow_single_node_partition)
        partitions = partitioner.propose_partitions()
        self.add_get_attr_inputs(partitions)
        fused_graph_module = partitioner.fuse_partitions(partitions)

        return fused_graph_module
