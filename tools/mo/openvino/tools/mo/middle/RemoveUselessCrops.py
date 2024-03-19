# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.middle.replacement import MiddleReplacementPattern


class RemoveUselessCropsPattern(MiddleReplacementPattern):
    r"""
    Remove useless construction with crops and concat like follows:
                in_node
         /    /   |    \     \
       crop crop  ..  crop  crop
        \    \    |    /    /
                out_node
    """
    enabled = True

    def run_after(self):
        from openvino.tools.mo.middle.RemoveDuplicationMemory import MergeNeighborSplicePattern
        return [MergeNeighborSplicePattern]

    @staticmethod
    def pattern():
        return dict(
            nodes=[('crop', dict(op='Crop')),
                   ('data', dict(kind='data')),
                   ('concat', dict(op='Concat'))],
            edges=[('crop', 'data'),
                   ('data', 'concat', {'in': 0})])

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        crop_node = match['crop']
        crop_node_parent_port = crop_node.in_port(0).get_source()
        concat_node = match['concat']

        if len(crop_node.out_port(0).get_destinations()) != 1:
            return

        outs = crop_node_parent_port.get_destinations()
        offsets_dims = list([])
        crop_list = list([])
        axis = crop_node['axis']
        for in_port in outs:
            out = in_port.node
            if out['op'] == 'Crop' and out['axis'] == axis and \
               len(out.out_port(0).get_destinations()) == 1 and \
               out.out_port(0).get_destination().node == concat_node:
                # crop type 1
                if 'dim' in out:
                    offsets_dims.append((out['offset'], out['dim']))
                # crop type 3
                elif 'crop_begin' in out and 'crop_end' in out:
                    offsets_dims.append((out['crop_begin'], out['crop_end']-out['crop_begin']))
                # crop type 2 with const dim
                elif not out.in_port(1).disconnected() and out.in_port(1).data.get_value() is not None:
                    offsets_dims.append((out['offset'], out.in_port(1).data.get_value()))
                # crop type 2 with non-const dim or strange type of crop
                else:
                    return
                crop_list.append(out)

        offsets_dims.sort(key=lambda off_dim: off_dim[0])
        size = 0
        for off_d in offsets_dims:
            if size != off_d[0]:
                return
            size = size + off_d[1]

        if size != crop_node_parent_port.data.get_shape()[axis]:
            return

        remove_concat = True
        free_port = None
        for inp in concat_node.in_ports():
            if not concat_node.in_port(inp).disconnected():
                in_node = concat_node.in_port(inp).get_source().node
                if in_node not in crop_list:
                    remove_concat = False
                else:
                    in_node.out_port(0).disconnect()
                    free_port = inp

        if remove_concat:
            concat_outs = concat_node.out_port(0).get_destinations()
            for out in concat_outs:
                out.disconnect()
                crop_node_parent_port.connect(out)
        else:
            crop_node_parent_port.connect(concat_node.in_port(free_port))
