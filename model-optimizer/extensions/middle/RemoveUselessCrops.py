"""
 Copyright (c) 2019 Intel Corporation

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
from mo.graph.graph import Graph, Node
from mo.middle.replacement import MiddleReplacementPattern


class RemoveUselessCropsPattern(MiddleReplacementPattern):
    """
    Remove useless construction with crops and concat like follows:
                in_node
         /    /   |    \     \
       crop crop  ..  crop  crop
        \    \    |    /    /
                out_node
    """
    enabled = False

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
        in_crop_node = crop_node.in_node(0)
        concat_node = match['concat']
        data = match['data']

        if len(data.out_nodes()) != 1:
            return

        outs = in_crop_node.out_nodes()
        offsets_dims = list([])
        crop_list = list([])
        axis = crop_node['axis']
        for out in outs:
            if out['op'] == 'Crop' and out['axis'] == axis and \
               len(out.out_node().out_nodes()) == 1 and out.out_node().out_node(0).id == concat_node.id:
                offsets_dims.append((out['offset'], out['dim']))
                crop_list.append(out.id)

        offsets_dims.sort(key=lambda off_dim: off_dim[0])
        size = 0
        for off_d in offsets_dims:
            if size != off_d[0]:
                return
            size = size + off_d[1]

        if size != in_crop_node.shape[axis]:
            return

        remove_concat = True
        for inp, attrs in concat_node.get_inputs():
            in_node_id, a = Node(graph, inp).get_inputs()[0]
            if in_node_id not in crop_list:
                remove_concat = False
            else:
                Node(graph, in_node_id).out_port(0).disconnect()

        if remove_concat:
            for crop in crop_list:
                Node(graph, crop).in_port(0).disconnect()

            concat_out = concat_node.out_node(0).out_node(0)
            concat_out.in_port(0).disconnect()
            in_crop_node.in_node(0).out_port(0).connect(concat_out.in_port(0))
