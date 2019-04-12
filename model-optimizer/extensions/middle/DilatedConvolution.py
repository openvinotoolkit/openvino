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
from mo.graph.graph import Graph
from mo.middle.replacement import MiddleReplacementPattern


class DilatedConvolutionConverter(MiddleReplacementPattern):
    enabled = True
    force_clean_up = True

    def run_after(self):
        from extensions.middle.pass_separator import PreMiddleStart
        return [PreMiddleStart]

    def run_before(self):
        from extensions.middle.pass_separator import MiddleStart
        return [MiddleStart]

    def pattern(self):
        return dict(
            nodes=[
                ('conv', dict(kind='op', op=lambda value: value in ['Conv2D', 'DepthwiseConv2dNative', 'Conv3D'])),
                ('space_to_batch', dict(kind='op', op='SpaceToBatchND')),
                ('batch_to_space', dict(kind='op', op='BatchToSpaceND')),
                ('input', dict(kind='data')),
                ('output', dict(kind='data')),
                ('conv_output', dict(kind='data')),
                ('stb_output', dict(kind='data')),
                ('stb_bs', dict(kind='data')),
                ('stb_pad', dict(kind='data')),
                ('bts_bs', dict(kind='data')),
                ('bts_crop', dict(kind='data'))
            ],
            edges=[
                ('input', 'space_to_batch', {'in': 0}),
                ('stb_bs', 'space_to_batch', {'in': 1}),
                ('stb_pad', 'space_to_batch', {'in': 2}),
                ('space_to_batch', 'stb_output', {'out': 0}),
                ('stb_output', 'conv', {'in': 0}),
                ('conv', 'conv_output', {'out': 0}),
                ('conv_output', 'batch_to_space', {'in': 0}),
                ('bts_bs', 'batch_to_space', {'in': 1}),
                ('bts_crop', 'batch_to_space', {'in': 2}),
                ('batch_to_space', 'output', {'out': 0}),
            ])

    def replace_pattern(self, graph: Graph, match: dict):
        conv = match['conv']
        stb = match['space_to_batch']
        bts = match['batch_to_space']

        block_size = match['stb_bs']

        input = match['input']
        output = match['output']
        stb_out = match['stb_output']
        conv_out = match['conv_output']

        in_edge_attrs = graph.get_edge_data(input.id, stb.id)[0]
        out_edge_attrs = graph.get_edge_data(bts.id, output.id)[0]

        graph.remove_edge(input.id, stb.id)
        graph.remove_edge(stb_out.id, conv.id)
        graph.remove_edge(conv.id, conv_out.id)
        graph.remove_edge(bts.id, output.id)

        conv.dilation[conv.spatial_dims] = block_size.value

        pad = match['stb_pad'].value - match['bts_crop'].value
        conv.pad[conv.spatial_dims] = [[pad[x][0], pad[x][1]] for x in range(len(pad))]
        conv['auto_pad'] = None

        graph.add_edges_from([
            (input.id, conv.id, {'in': 0, **in_edge_attrs}),
            (conv.id, output.id, {'out': 0, **out_edge_attrs}),
        ])
