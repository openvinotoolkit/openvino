"""
 Copyright (c) 2020 Intel Corporation

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


import unittest

from extensions.middle.InterpolateSequenceToInterpolate import InterpolateSequenceToInterpolate
from mo.front.common.partial_infer.utils import int64_array
from mo.utils.unittest.graph import build_graph
from mo.utils.ir_engine.compare_graphs import compare_graphs


graph_node_attrs_for_2d_case_1 = {
    'placeholder': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_data': {
        'value': None,
        'shape': int64_array([1, 4, 220, 350]),
        'kind': 'data',
        'data_type': None
    },
    'scale_1': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([660])
    },
    'scale_1_data': {'value': None, 'shape': [1], 'kind': 'data'},
    'interpolate_1': {
        'type': 'Interpolate',
        'kind': 'op',
        'op': 'Interpolate',
        'axes': int64_array([2]),
        'mode': 'nearest'
    },
    'interpolate_1_data': {'value': None, 'shape': int64_array([1, 4, 660, 350]), 'kind': 'data'},
    'scale_2': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([700])
    },
    'scale_2_data': {'value': None, 'shape': [1], 'kind': 'data'},
    'interpolate_2': {
        'type': 'Interpolate',
        'kind': 'op',
        'op': 'Interpolate',
        'axes': int64_array([3]),
        'mode': 'nearest'
    },
    'interpolate_2_data': {'value': None, 'shape': int64_array([1, 4, 660, 700]), 'kind': 'data'},
    'scale_3': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([1320])
    },
    'scale_3_data': {'value': None, 'shape': [1], 'kind': 'data'},
    'interpolate_3': {
        'type': 'Interpolate',
        'kind': 'op',
        'op': 'Interpolate',
        'axes': int64_array([2]),
        'mode': 'nearest'
    },
    'interpolate_3_data': {'value': None, 'shape': int64_array([1, 4, 1320, 700]), 'kind': 'data'},
    'abs': {'type': 'Abs', 'kind': 'op', 'op': 'Abs'},
    'abs_data': {'value': None, 'shape': int64_array([1, 4, 1320, 700]), 'kind': 'data'},
    'output': {'kind': 'op', 'op': 'Result'},
}

edges_for_2d_case_1 = [
    ('placeholder', 'placeholder_data'),

    ('placeholder_data', 'interpolate_1', {'in': 0}),
    ('scale_1', 'scale_1_data'),
    ('scale_1_data', 'interpolate_1', {'in': 1}),
    ('interpolate_1', 'interpolate_1_data'),

    ('interpolate_1_data', 'interpolate_2', {'in': 0}),
    ('scale_2', 'scale_2_data'),
    ('scale_2_data', 'interpolate_2', {'in': 1}),
    ('interpolate_2', 'interpolate_2_data'),

    ('interpolate_2_data', 'interpolate_3', {'in': 0}),
    ('scale_3', 'scale_3_data'),
    ('scale_3_data', 'interpolate_3', {'in': 1}),
    ('interpolate_3', 'interpolate_3_data'),

    ('interpolate_3_data', 'abs'),
    ('abs', 'abs_data'),
    ('abs_data', 'output'),
]


graph_node_attrs_for_2d_case_2 = {
    'placeholder': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_data': {
        'value': None,
        'shape': int64_array([1, 4, 220, 350]),
        'kind': 'data',
        'data_type': None
    },
    'scale_1': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([660])
    },
    'scale_1_data': {'value': None, 'shape': [1], 'kind': 'data'},
    'interpolate_1': {
        'type': 'Interpolate',
        'kind': 'op',
        'op': 'Interpolate',
        'axes': int64_array([2]),
        'mode': 'nearest'
    },
    'interpolate_1_data': {'value': None, 'shape': int64_array([1, 4, 660, 350]), 'kind': 'data'},
    'abs': {'type': 'Abs', 'kind': 'op', 'op': 'Abs'},
    'abs_data': {'value': None, 'shape': int64_array([1, 4, 660, 350]), 'kind': 'data'},
    'output': {'kind': 'op', 'op': 'Result'},
}

edges_for_2d_case_2 = [
    ('placeholder', 'placeholder_data'),

    ('placeholder_data', 'interpolate_1', {'in': 0}),
    ('scale_1', 'scale_1_data'),
    ('scale_1_data', 'interpolate_1', {'in': 1}),
    ('interpolate_1', 'interpolate_1_data'),

    ('interpolate_1_data', 'abs'),
    ('abs', 'abs_data'),
    ('abs_data', 'output'),
]


graph_node_attrs_for_2d_case_3 = {
    'placeholder': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_data': {
        'value': None,
        'shape': int64_array([1, 4, 220, 350]),
        'kind': 'data',
        'data_type': None
    },
    'scale_1': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([660])
    },
    'scale_1_data': {'value': None, 'shape': [1], 'kind': 'data'},
    'interpolate_1': {
        'type': 'Interpolate',
        'kind': 'op',
        'op': 'Interpolate',
        'axes': int64_array([2]),
        'mode': 'nearest'
    },
    'interpolate_1_data': {'value': None, 'shape': int64_array([1, 4, 660, 350]), 'kind': 'data'},
    'scale_2': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([700])
    },
    'scale_2_data': {'value': None, 'shape': [1], 'kind': 'data'},
    'interpolate_2': {
        'type': 'Interpolate',
        'kind': 'op',
        'op': 'Interpolate',
        'axes': int64_array([3]),
        'mode': 'linear'
    },
    'interpolate_2_data': {'value': None, 'shape': int64_array([1, 4, 660, 700]), 'kind': 'data'},
    'scale_3': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([1320])
    },
    'scale_3_data': {'value': None, 'shape': [1], 'kind': 'data'},
    'interpolate_3': {
        'type': 'Interpolate',
        'kind': 'op',
        'op': 'Interpolate',
        'axes': int64_array([2]),
        'mode': 'cubic'
    },
    'interpolate_3_data': {'value': None, 'shape': int64_array([1, 4, 1320, 700]), 'kind': 'data'},
    'abs': {'type': 'Abs', 'kind': 'op', 'op': 'Abs'},
    'abs_data': {'value': None, 'shape': int64_array([1, 4, 1320, 700]), 'kind': 'data'},
    'output': {'kind': 'op', 'op': 'Result'},
}

edges_for_2d_case_3 = edges_for_2d_case_1


graph_node_attrs_for_2d_case_4 = {
    'placeholder': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_data': {
        'value': None,
        'shape': int64_array([1, 4, 220, 350]),
        'kind': 'data',
        'data_type': None
    },
    'scale_1': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([2200])
    },
    'scale_1_data': {'value': None, 'shape': [1], 'kind': 'data'},
    'interpolate_1': {
        'type': 'Interpolate',
        'kind': 'op',
        'op': 'Interpolate',
        'axes': int64_array([2]),
        'mode': 'linear',
        'align_corners': 0,
        'antialias': 1,
        'pads_begin': 5,
        'pads_end': 3
    },
    'interpolate_1_data': {'value': None, 'shape': int64_array([1, 4, 2200, 350]), 'kind': 'data'},
    'scale_2': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([700])
    },
    'scale_2_data': {'value': None, 'shape': [1], 'kind': 'data'},
    'interpolate_2': {
        'type': 'Interpolate',
        'kind': 'op',
        'op': 'Interpolate',
        'axes': int64_array([3]),
        'mode': 'linear',
        'align_corners': 0,
        'antialias': 1,
        'pads_begin': 5,
        'pads_end': 3
    },
    'interpolate_2_data': {'value': None, 'shape': int64_array([1, 4, 2200, 700]), 'kind': 'data'},
    'abs': {'type': 'Abs', 'kind': 'op', 'op': 'Abs'},
    'abs_data': {'value': None, 'shape': int64_array([1, 4, 2200, 700]), 'kind': 'data'},
    'output': {'kind': 'op', 'op': 'Result'},
}

edges_for_2d_case_4 = [
    ('placeholder', 'placeholder_data'),

    ('placeholder_data', 'interpolate_1', {'in': 0}),
    ('scale_1', 'scale_1_data'),
    ('scale_1_data', 'interpolate_1', {'in': 1}),
    ('interpolate_1', 'interpolate_1_data'),

    ('interpolate_1_data', 'interpolate_2', {'in': 0}),
    ('scale_2', 'scale_2_data'),
    ('scale_2_data', 'interpolate_2', {'in': 1}),
    ('interpolate_2', 'interpolate_2_data'),

    ('interpolate_2_data', 'abs'),
    ('abs', 'abs_data'),
    ('abs_data', 'output'),
]


graph_node_attrs_for_2d_case_6 = {
    'placeholder': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_data': {
        'value': None,
        'shape': int64_array([1, 4, 220, 350]),
        'kind': 'data',
        'data_type': None
    },
    'scale_1': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([220, 350])
    },
    'scale_1_data': {'value': None, 'shape': [2], 'kind': 'data'},
    'interpolate_1': {
        'type': 'Interpolate',
        'kind': 'op',
        'op': 'Interpolate',
        'axes': int64_array([2, 3]),
        'mode': 'linear',
        'align_corners': 0,
        'antialias': 1,
        'pads_begin': 5,
        'pads_end': 3
    },
    'interpolate_1_data': {'value': None, 'shape': int64_array([1, 4, 220, 350]), 'kind': 'data'},
    'scale_2': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([220])
    },
    'scale_2_data': {'value': None, 'shape': [1], 'kind': 'data'},
    'interpolate_2': {
        'type': 'Interpolate',
        'kind': 'op',
        'op': 'Interpolate',
        'axes': int64_array([2]),
        'mode': 'linear',
        'align_corners': 0,
        'antialias': 1,
        'pads_begin': 5,
        'pads_end': 3
    },
    'interpolate_2_data': {'value': None, 'shape': int64_array([1, 4, 220, 350]), 'kind': 'data'},
    'abs': {'type': 'Abs', 'kind': 'op', 'op': 'Abs'},
    'abs_data': {'value': None, 'shape': int64_array([1, 4, 220, 350]), 'kind': 'data'},
    'output': {'kind': 'op', 'op': 'Result'},
}

edges_for_2d_case_6 = edges_for_2d_case_4


graph_node_attrs_for_3d_case_1 = {
    'placeholder': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_data': {
        'value': None,
        'shape': int64_array([1, 5, 1024, 256, 800]),
        'kind': 'data',
        'data_type': None
    },
    'scale_1': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([4096, 2400])
    },
    'scale_1_data': {'value': None, 'shape': [2], 'kind': 'data'},
    'interpolate_1': {
        'type': 'Interpolate',
        'kind': 'op',
        'op': 'Interpolate',
        'axes': int64_array([2, 4]),
        'mode': 'nearest'
    },
    'interpolate_1_data': {'value': None, 'shape': int64_array([1, 5, 4096, 256, 2400]), 'kind': 'data'},
    'scale_2': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([1280])
    },
    'scale_2_data': {'value': None, 'shape': [1], 'kind': 'data'},
    'interpolate_2': {
        'type': 'Interpolate',
        'kind': 'op',
        'op': 'Interpolate',
        'axes': int64_array([3]),
        'mode': 'nearest'
    },
    'interpolate_2_data': {'value': None, 'shape': int64_array([1, 5, 4096, 1280, 2400]), 'kind': 'data'},
    'scale_3': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([512])
    },
    'scale_3_data': {'value': None, 'shape': [1], 'kind': 'data'},
    'interpolate_3': {
        'type': 'Interpolate',
        'kind': 'op',
        'op': 'Interpolate',
        'axes': int64_array([4]),
        'mode': 'nearest'
    },
    'interpolate_3_data': {'value': None, 'shape': int64_array([1, 5, 4096, 1280, 512]), 'kind': 'data'},
    'abs': {'type': 'Abs', 'kind': 'op', 'op': 'Abs'},
    'abs_data': {'value': None, 'shape': int64_array([1, 5, 4096, 1280, 512]), 'kind': 'data'},
    'output': {'kind': 'op', 'op': 'Result'},
}

edges_for_3d_case_1 = edges_for_2d_case_1


graph_node_attrs_for_3d_case_2 = {
    'placeholder': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_data': {
        'value': None,
        'shape': int64_array([1, 5, 1024, 256, 800]),
        'kind': 'data',
        'data_type': None
    },
    'scale_1': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([4096, 1280])
    },
    'scale_1_data': {'value': None, 'shape': [1], 'kind': 'data'},
    'interpolate_1': {
        'type': 'Interpolate',
        'kind': 'op',
        'op': 'Interpolate',
        'axes': int64_array([2, 3]),
        'mode': 'nearest'
    },
    'interpolate_1_data': {'value': None, 'shape': int64_array([1, 5, 4096, 1280, 800]), 'kind': 'data'},
    'abs': {'type': 'Abs', 'kind': 'op', 'op': 'Abs'},
    'abs_data': {'value': None, 'shape': int64_array([1, 5, 4096, 1280, 800]), 'kind': 'data'},
    'output': {'kind': 'op', 'op': 'Result'},
}

edges_for_3d_case_2 = edges_for_2d_case_2


graph_node_attrs_for_3d_case_3 = {
    'placeholder': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_data': {
        'value': None,
        'shape': int64_array([16, 44, 512, 87, 790]),
        'kind': 'data',
        'data_type': None
    },
    'scale_1': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([256])
    },
    'scale_1_data': {'value': None, 'shape': [1], 'kind': 'data'},
    'interpolate_1': {
        'type': 'Interpolate',
        'kind': 'op',
        'op': 'Interpolate',
        'axes': int64_array([2]),
        'mode': 'nearest'
    },
    'interpolate_1_data': {'value': None, 'shape': int64_array([16, 44, 256, 87, 790]), 'kind': 'data'},
    'scale_2': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([2370])
    },
    'scale_2_data': {'value': None, 'shape': [1], 'kind': 'data'},
    'interpolate_2': {
        'type': 'Interpolate',
        'kind': 'op',
        'op': 'Interpolate',
        'axes': int64_array([4]),
        'mode': 'linear'
    },
    'interpolate_2_data': {'value': None, 'shape': int64_array([16, 44, 256, 87, 2370]), 'kind': 'data'},
    'scale_3': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([435])
    },
    'scale_3_data': {'value': None, 'shape': [1], 'kind': 'data'},
    'interpolate_3': {
        'type': 'Interpolate',
        'kind': 'op',
        'op': 'Interpolate',
        'axes': int64_array([3]),
        'mode': 'cubic'
    },
    'interpolate_3_data': {'value': None, 'shape': int64_array([16, 44, 256, 435, 2370]), 'kind': 'data'},
    'abs': {'type': 'Abs', 'kind': 'op', 'op': 'Abs'},
    'abs_data': {'value': None, 'shape': int64_array([16, 44, 256, 435, 2370]), 'kind': 'data'},
    'output': {'kind': 'op', 'op': 'Result'},
}

edges_for_3d_case_3 = edges_for_2d_case_3


graph_node_attrs_for_3d_case_4 = {
    'placeholder': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_data': {
        'value': None,
        'shape': int64_array([10, 64, 511, 416, 10240]),
        'kind': 'data',
        'data_type': None
    },
    'scale_1': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([4599, 133120])
    },
    'scale_1_data': {'value': None, 'shape': [2], 'kind': 'data'},
    'interpolate_1': {
        'type': 'Interpolate',
        'kind': 'op',
        'op': 'Interpolate',
        'axes': int64_array([2, 4]),
        'mode': 'linear',
        'align_corners': 0,
        'antialias': 1,
        'pads_begin': 5,
        'pads_end': 3
    },
    'interpolate_1_data': {'value': None, 'shape': int64_array([10, 64, 4599, 416, 133120]), 'kind': 'data'},
    'scale_2': {
        'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([912])
    },
    'scale_2_data': {'value': None, 'shape': [1], 'kind': 'data'},
    'interpolate_2': {
        'type': 'Interpolate',
        'kind': 'op',
        'op': 'Interpolate',
        'axes': int64_array([3]),
        'mode': 'linear',
        'align_corners': 0,
        'antialias': 1,
        'pads_begin': 5,
        'pads_end': 3
    },
    'interpolate_2_data': {'value': None, 'shape': int64_array([10, 64, 4599, 912, 133120]), 'kind': 'data'},
    'abs': {'type': 'Abs', 'kind': 'op', 'op': 'Abs'},
    'abs_data': {'value': None, 'shape': int64_array([10, 64, 4599, 912, 133120]), 'kind': 'data'},
    'output': {'kind': 'op', 'op': 'Result'},
}

edges_for_3d_case_4 = edges_for_2d_case_4


class InterpolateSequenceToInterpolateTest(unittest.TestCase):
    def test_2d_interpolate_sequence_1(self):
        graph = build_graph(
            nodes_attrs=graph_node_attrs_for_2d_case_1,
            edges=edges_for_2d_case_1
        )

        ref_graph = build_graph(
            nodes_attrs={
                'placeholder': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
                'placeholder_data': {
                    'value': None,
                    'shape': int64_array([1, 4, 220, 350]),
                    'kind': 'data',
                    'data_type': None
                },
                'scale_1': {
                    'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([660, 700])
                },
                'scale_1_data': {'value': None, 'shape': None, 'kind': 'data'},
                'interpolate_1': {
                    'type': 'Interpolate',
                    'kind': 'op',
                    'op': 'Interpolate',
                    'axes': int64_array([2, 3]),
                    'mode': 'nearest'
                },
                'interpolate_1_data': {'value': None, 'shape': int64_array([1, 4, 660, 700]), 'kind': 'data'},
                'scale_2': {
                    'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([1320])
                },
                'scale_2_data': {'value': None, 'shape': None, 'kind': 'data'},
                'interpolate_2': {
                    'type': 'Interpolate',
                    'kind': 'op',
                    'op': 'Interpolate',
                    'axes': int64_array([2]),
                    'mode': 'nearest'
                },
                'interpolate_2_data': {'value': None, 'shape': int64_array([1, 4, 1320, 700]), 'kind': 'data'},
                'abs': {'type': 'Abs', 'kind': 'op', 'op': 'Abs'},
                'abs_data': {'value': None, 'shape': int64_array([1, 4, 1320, 700]), 'kind': 'data'},
                'output': {'kind': 'op', 'op': 'Result'},
            },
            edges=[
                ('placeholder', 'placeholder_data'),
                ('placeholder_data', 'interpolate_1', {'in': 0}),
                ('scale_1', 'scale_1_data'),
                ('scale_1_data', 'interpolate_1', {'in': 1}),
                ('interpolate_1', 'interpolate_1_data'),
                ('scale_2', 'scale_2_data'),
                ('interpolate_2', 'interpolate_2_data'),
                ('interpolate_1_data', 'interpolate_2', {'in': 0}),
                ('scale_2_data', 'interpolate_2', {'in': 1}),
                ('interpolate_2_data', 'abs'),
                ('abs', 'abs_data'),
                ('abs_data', 'output'),
            ]
        )
        InterpolateSequenceToInterpolate().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, ref_graph, 'output')
        self.assertTrue(flag, resp)

    def test_2d_interpolate_sequence_2(self):
        graph = build_graph(
            nodes_attrs=graph_node_attrs_for_2d_case_2,
            edges=edges_for_2d_case_2
        )
        ref_graph = build_graph(
            nodes_attrs=graph_node_attrs_for_2d_case_2,
            edges=edges_for_2d_case_2
        )
        InterpolateSequenceToInterpolate().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, ref_graph, 'output')
        self.assertTrue(flag, resp)

    def test_2d_interpolate_sequence_3(self):
        graph = build_graph(
            nodes_attrs=graph_node_attrs_for_2d_case_3,
            edges=edges_for_2d_case_3
        )

        ref_graph = build_graph(
            nodes_attrs=graph_node_attrs_for_2d_case_3,
            edges=edges_for_2d_case_3
        )

        InterpolateSequenceToInterpolate().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, ref_graph, 'output')
        self.assertTrue(flag, resp)

    def test_2d_interpolate_sequence_4(self):
        graph = build_graph(
            nodes_attrs=graph_node_attrs_for_2d_case_4,
            edges=edges_for_2d_case_4
        )

        ref_graph = build_graph(
            nodes_attrs={
                'placeholder': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
                'placeholder_data': {
                    'value': None,
                    'shape': int64_array([1, 4, 220, 350]),
                    'kind': 'data',
                    'data_type': None
                },
                'scale': {
                    'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([2200, 700])
                },
                'scale_data': {'value': None, 'shape': None, 'kind': 'data'},
                'interpolate': {
                    'type': 'Interpolate',
                    'kind': 'op',
                    'op': 'Interpolate',
                    'axes': int64_array([2, 3]),
                    'mode': 'linear',
                    'align_corners': 0,
                    'antialias': 1,
                    'pads_begin': 5,
                    'pads_end': 3
                },
                'interpolate_data': {'value': None, 'shape': int64_array([1, 4, 2200, 700]), 'kind': 'data'},
                'abs': {'type': 'Abs', 'kind': 'op', 'op': 'Abs'},
                'abs_data': {'value': None, 'shape': int64_array([1, 4, 2200, 700]), 'kind': 'data'},
                'output': {'kind': 'op', 'op': 'Result'},
            },
            edges=[
                ('placeholder', 'placeholder_data'),

                ('placeholder_data', 'interpolate', {'in': 0}),
                ('scale', 'scale_data'),
                ('scale_data', 'interpolate', {'in': 1}),
                ('interpolate', 'interpolate_data'),

                ('interpolate_data', 'abs'),
                ('abs', 'abs_data'),
                ('abs_data', 'output'),
            ]
        )

        InterpolateSequenceToInterpolate().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, ref_graph, 'output')
        self.assertTrue(flag, resp)

    def test_2d_interpolate_sequence_5(self):
        graph = build_graph(
            nodes_attrs=graph_node_attrs_for_2d_case_4,
            edges=edges_for_2d_case_4,
            update_attributes={
                'interpolate_1': {
                    'align_corners': 1, 'antialias': 1, 'pads_begin': 3, 'pads_end': 0
                }
            }
        )

        ref_graph = build_graph(
            nodes_attrs=graph_node_attrs_for_2d_case_4,
            edges=edges_for_2d_case_4,
            update_attributes={
                'interpolate_1': {
                    'align_corners': 1, 'antialias': 1, 'pads_begin': 3, 'pads_end': 0
                }
            }
        )

        InterpolateSequenceToInterpolate().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, ref_graph, 'output')
        self.assertTrue(flag, resp)

    def test_2d_interpolate_sequence_6(self):
        graph = build_graph(
            nodes_attrs=graph_node_attrs_for_2d_case_6,
            edges=edges_for_2d_case_6,
        )

        ref_graph = build_graph(
            nodes_attrs=graph_node_attrs_for_2d_case_6,
            edges=edges_for_2d_case_6
        )

        InterpolateSequenceToInterpolate().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, ref_graph, 'output')
        self.assertTrue(flag, resp)

    def test_3d_interpolate_sequence_1(self):
        graph = build_graph(
            nodes_attrs=graph_node_attrs_for_3d_case_1,
            edges=edges_for_3d_case_1
        )

        ref_graph = build_graph(
            nodes_attrs={
                'placeholder': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
                'placeholder_data': {
                    'value': None,
                    'shape': int64_array([1, 5, 1024, 256, 800]),
                    'kind': 'data',
                    'data_type': None
                },
                'scale_1': {
                    'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([4096, 1280, 2400])
                },
                'scale_1_data': {'value': None, 'shape': None, 'kind': 'data'},
                'interpolate_1': {
                    'type': 'Interpolate',
                    'kind': 'op',
                    'op': 'Interpolate',
                    'axes': int64_array([2, 3, 4]),
                    'mode': 'nearest'
                },
                'interpolate_1_data': {'value': None, 'shape': int64_array([1, 5, 4096, 1280, 2400]), 'kind': 'data'},
                'scale_2': {
                    'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([512])
                },
                'scale_2_data': {'value': None, 'shape': [1], 'kind': 'data'},
                'interpolate_2': {
                    'type': 'Interpolate',
                    'kind': 'op',
                    'op': 'Interpolate',
                    'axes': int64_array([4]),
                    'mode': 'nearest'
                },
                'interpolate_2_data': {'value': None, 'shape': int64_array([1, 5, 4096, 1280, 512]), 'kind': 'data'},
                'abs': {'type': 'Abs', 'kind': 'op', 'op': 'Abs'},
                'abs_data': {'value': None, 'shape': int64_array([1, 5, 4096, 1280, 512]), 'kind': 'data'},
                'output': {'kind': 'op', 'op': 'Result'},
            },
            edges=[
                ('placeholder', 'placeholder_data'),
                ('placeholder_data', 'interpolate_1', {'in': 0}),
                ('scale_1', 'scale_1_data'),
                ('scale_1_data', 'interpolate_1', {'in': 1}),
                ('interpolate_1', 'interpolate_1_data'),
                ('scale_2', 'scale_2_data'),
                ('interpolate_2', 'interpolate_2_data'),
                ('interpolate_1_data', 'interpolate_2', {'in': 0}),
                ('scale_2_data', 'interpolate_2', {'in': 1}),
                ('interpolate_2_data', 'abs'),
                ('abs', 'abs_data'),
                ('abs_data', 'output'),
            ]
        )
        InterpolateSequenceToInterpolate().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, ref_graph, 'output')
        self.assertTrue(flag, resp)

    def test_3d_interpolate_sequence_2(self):
        graph = build_graph(
            nodes_attrs=graph_node_attrs_for_3d_case_2,
            edges=edges_for_3d_case_2
        )
        ref_graph = build_graph(
            nodes_attrs=graph_node_attrs_for_3d_case_2,
            edges=edges_for_3d_case_2
        )
        InterpolateSequenceToInterpolate().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, ref_graph, 'output')
        self.assertTrue(flag, resp)

    def test_3d_interpolate_sequence_3(self):
        graph = build_graph(
            nodes_attrs=graph_node_attrs_for_3d_case_3,
            edges=edges_for_3d_case_3
        )
        ref_graph = build_graph(
            nodes_attrs=graph_node_attrs_for_3d_case_3,
            edges=edges_for_3d_case_3
        )
        InterpolateSequenceToInterpolate().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, ref_graph, 'output')
        self.assertTrue(flag, resp)

    def test_3d_interpolate_sequence_4(self):
        graph = build_graph(
            nodes_attrs=graph_node_attrs_for_3d_case_4,
            edges=edges_for_3d_case_4
        )

        ref_graph = build_graph(
            nodes_attrs={
                'placeholder': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
                'placeholder_data': {
                    'value': None,
                    'shape': int64_array([10, 64, 511, 416, 10240]),
                    'kind': 'data',
                    'data_type': None
                },
                'scale': {
                    'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': int64_array([4599, 912, 133120])
                },
                'scale_data': {'value': None, 'shape': None, 'kind': 'data'},
                'interpolate': {
                    'type': 'Interpolate',
                    'kind': 'op',
                    'op': 'Interpolate',
                    'axes': int64_array([2, 3, 4]),
                    'mode': 'linear',
                    'align_corners': 0,
                    'antialias': 1,
                    'pads_begin': 5,
                    'pads_end': 3
                },
                'interpolate_data': {'value': None, 'shape': int64_array([10, 64, 4599, 912, 133120]), 'kind': 'data'},
                'abs': {'type': 'Abs', 'kind': 'op', 'op': 'Abs'},
                'abs_data': {'value': None, 'shape': int64_array([10, 64, 4599, 912, 133120]), 'kind': 'data'},
                'output': {'kind': 'op', 'op': 'Result'},
            },
            edges=[
                ('placeholder', 'placeholder_data'),

                ('placeholder_data', 'interpolate', {'in': 0}),
                ('scale', 'scale_data'),
                ('scale_data', 'interpolate', {'in': 1}),
                ('interpolate', 'interpolate_data'),

                ('interpolate_data', 'abs'),
                ('abs', 'abs_data'),
                ('abs_data', 'output'),
            ]
        )

        InterpolateSequenceToInterpolate().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, ref_graph, 'output')
        self.assertTrue(flag, resp)

    def test_3d_interpolate_sequence_5(self):
        graph = build_graph(
            nodes_attrs=graph_node_attrs_for_3d_case_4,
            edges=edges_for_3d_case_4,
            update_attributes={
                'interpolate_1': {
                    'align_corners': 1, 'antialias': 1, 'pads_begin': 3, 'pads_end': 7
                }
            }
        )

        ref_graph = build_graph(
            nodes_attrs=graph_node_attrs_for_3d_case_4,
            edges=edges_for_3d_case_4,
            update_attributes={
                'interpolate_1': {
                    'align_corners': 1, 'antialias': 1, 'pads_begin': 3, 'pads_end': 7
                }
            }
        )

        InterpolateSequenceToInterpolate().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, ref_graph, 'output')
        self.assertTrue(flag, resp)
