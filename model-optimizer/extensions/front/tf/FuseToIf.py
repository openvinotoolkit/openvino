# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from extensions.ops.loop import Loop
from mo.front.common.replacement import FrontReplacementSubgraph
from mo.graph.graph import Graph, Node
from mo.ops.const import Const


class FuseToIF(FrontReplacementSubgraph):
    """
    Find Switch/Merge Nodes and fuse it to If node
    """
    enabled = True

    def find_and_replace_pattern(self, graph: Graph):
        pass
        for node in graph.get_op_nodes(op='Merge'):
            if len(node.in_ports(True))==2:
                self.fuse_to_if(node)

    @staticmethod
    def fuse_to_if(merge_node):
        switch_nodes={}
        merge_nodes = {}
        merge_nodes[merge_node.soft_get('name')] = merge_node
        in_nodes_list=[]
        target_node


    def find_switches(self):
        inputs_list =[]
