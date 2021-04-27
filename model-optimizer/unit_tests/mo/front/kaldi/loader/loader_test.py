# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import io
import struct
import unittest

import numpy as np

from mo.front.kaldi.loader.loader import load_topology_map, load_components
from mo.graph.graph import Graph, Node
from mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph


class TestKaldiModelsLoading(unittest.TestCase):

    def test_component_map_loading_sequence(self):
        test_map = "input-node name=input dim=16 \n" + \
                   "component-node name=lda component=lda input=input \n" + \
                   "component-node name=tdnn1.affine component=tdnn1.affine input=lda \n" + \
                   "component-node name=tdnn1.relu component=tdnn1.relu input=tdnn1.affine \n" + \
                   "component-node name=tdnn1.batchnorm component=tdnn1.batchnorm input=tdnn1.relu \n\n"
        graph = Graph(name="test_graph_component_map_loading_sequence")

        test_top_map = load_topology_map(io.BytesIO(bytes(test_map, 'ascii')), graph)

        ref_map = {b"lda": ["lda"],
                   b"tdnn1.affine": ["tdnn1.affine"],
                   b"tdnn1.relu": ["tdnn1.relu"],
                   b"tdnn1.batchnorm": ["tdnn1.batchnorm"]}
        self.assertEqual(test_top_map, ref_map)
        self.assertTrue("input" in graph.nodes())
        self.assertListEqual(list(Node(graph, 'input')['shape']), [1, 16])

        ref_graph = build_graph({'input': {'shape': np.array([1, 16]), 'kind': 'op', 'op': 'Parameter'},
                                 'lda': {'kind': 'op'},
                                 'tdnn1.affine': {'kind': 'op'},
                                 'tdnn1.relu': {'kind': 'op'},
                                 'tdnn1.batchnorm': {'kind': 'op'},
                                 },
                                [
                                    ('input', 'lda'),
                                    ('lda', 'tdnn1.affine'),
                                    ('tdnn1.affine', 'tdnn1.relu'),
                                    ('tdnn1.relu', 'tdnn1.batchnorm'),
                                ]
                                )
        (flag, resp) = compare_graphs(graph, ref_graph, 'tdnn1.batchnorm')
        self.assertTrue(flag, resp)

    # NOTE: this test is disabled because it's broken and need to be fixed! Merge request 948.
    # Fail in load_topology_map() in read_node() method - we create edge with node which doesn't exist in graph
    def test_component_map_loading_swap(self):
        test_map = "input-node name=input dim=16 \n" + \
                   "component-node name=lda component=lda input=input \n" + \
                   "component-node name=tdnn1.batchnorm component=tdnn1.batchnorm input=tdnn1.relu \n" + \
                   "component-node name=tdnn1.relu component=tdnn1.relu input=tdnn1.affine \n" + \
                   "component-node name=tdnn1.affine component=tdnn1.affine input=lda \n" + \
                   "\n"
        graph = Graph(name="test_graph_component_map_loading_swap")

        test_top_map = load_topology_map(io.BytesIO(bytes(test_map, 'ascii')), graph)

        ref_map = {b"lda": ["lda"],
                   b"tdnn1.affine": ["tdnn1.affine"],
                   b"tdnn1.relu": ["tdnn1.relu"],
                   b"tdnn1.batchnorm": ["tdnn1.batchnorm"]}
        self.assertEqual(test_top_map, ref_map)
        self.assertTrue("input" in graph.nodes())
        self.assertListEqual(list(Node(graph, 'input')['shape']), [1, 16])

        ref_graph = build_graph({'input': {'shape': np.array([1, 16]), 'kind': 'op', 'op': 'Parameter'},
                                 'lda': {'kind': 'op'},
                                 'tdnn1.affine': {'kind': 'op'},
                                 'tdnn1.relu': {'kind': 'op'},
                                 'tdnn1.batchnorm': {'kind': 'op'},
                                 },
                                [
                                    ('input', 'lda'),
                                    ('lda', 'tdnn1.affine'),
                                    ('tdnn1.affine', 'tdnn1.relu'),
                                    ('tdnn1.relu', 'tdnn1.batchnorm'),
                                ]
                                )
        (flag, resp) = compare_graphs(graph, ref_graph, 'tdnn1.batchnorm')
        self.assertTrue(flag, resp)

    def test_component_map_loading_append(self):
        test_map = "input-node name=input dim=16 \n" + \
                   "component-node name=lda component=lda input=input \n" + \
                   "component-node name=tdnn1.affine component=tdnn1.affine input=Append(input, lda) \n" + \
                   "component-node name=tdnn1.relu component=tdnn1.relu input=Append(tdnn1.affine, input, lda) \n" + \
                   "\n"
        graph = Graph(name="test_graph_component_map_loading_append")

        test_top_map= load_topology_map(io.BytesIO(bytes(test_map, 'ascii')), graph)

        ref_map = {b"lda": ["lda"],
                   b"tdnn1.affine": ["tdnn1.affine"],
                   b"tdnn1.relu": ["tdnn1.relu"]}
        self.assertEqual(test_top_map, ref_map)
        self.assertTrue("input" in graph.nodes())
        self.assertListEqual(list(Node(graph, 'input')['shape']), [1, 16])

        ref_graph = build_graph({'input': {'shape': np.array([1, 16]), 'kind': 'op', 'op': 'Parameter'},
                                 'lda': {'kind': 'op'},
                                 'tdnn1.affine': {'kind': 'op'},
                                 'tdnn1.relu': {'kind': 'op'},
                                 'append_input_lda': {'kind': 'op', 'op': 'Concat'},
                                 'append_affine_input_lda': {'kind': 'op', 'op': 'Concat'},
                                 },
                                [
                                    ('input', 'lda', {'out': 0}),
                                    ('lda', 'append_input_lda', {'in': 1, 'out': 0}),
                                    ('input', 'append_input_lda', {'in': 0, 'out': 1}),
                                    ('append_input_lda', 'tdnn1.affine', {'out': 0}),
                                    ('input', 'append_affine_input_lda', {'in': 1, 'out': 2}),
                                    ('lda', 'append_affine_input_lda', {'in': 2, 'out': 1}),
                                    ('tdnn1.affine', 'append_affine_input_lda', {'in': 0, 'out': 0}),
                                    ('append_affine_input_lda', 'tdnn1.relu', {'out': 0}),
                                ]
                                )

        (flag, resp) = compare_graphs(graph, ref_graph, 'tdnn1.relu')
        self.assertTrue(flag, resp)

    def test_component_map_loading_offset(self):
        test_map = "input-node name=input dim=16\n" + \
                   "component-node name=lda component=lda input=Offset(input, -3)\n" + \
                   "component-node name=tdnn1.affine component=tdnn1.affine input=Append(Offset(input, -1), Offset(lda, 1))\n" + \
                   "component-node name=tdnn1.relu component=tdnn1.relu input=tdnn1.affine\n" + \
                   "\n"
        graph = Graph(name="test_graph_component_map_loading_offset")

        test_top_map= load_topology_map(io.BytesIO(bytes(test_map, 'ascii')), graph)

        ref_map = {b"lda": ["lda"],
                   b"tdnn1.affine": ["tdnn1.affine"],
                   b"tdnn1.relu": ["tdnn1.relu"]}
        self.assertEqual(test_top_map, ref_map)
        self.assertTrue("input" in graph.nodes())
        self.assertListEqual(list(Node(graph, 'input')['shape']), [1, 16])

        ref_graph = build_graph({'input': {'shape': np.array([1, 16]), 'kind': 'op', 'op': 'Parameter'},
                                 'lda': {'kind': 'op'},
                                 'tdnn1.affine': {'kind': 'op'},
                                 'tdnn1.relu': {'kind': 'op'},
                                 'append_input_lda': {'kind': 'op', 'op': 'Concat'},
                                 'offset_in_input_3': {'kind': 'op', 'op': 'memoryoffset', 't': -3, 'pair_name': 'offset_out_input_3'},
                                 'offset_in_input_1': {'kind': 'op', 'op': 'memoryoffset', 't': -1, 'pair_name': 'offset_out_input_1'},
                                 'offset_in_lda_1': {'kind': 'op', 'op': 'memoryoffset', 't': -1, 'pair_name': 'offset_out_lda_1'},
                                 },
                                [
                                    ('input', 'offset_in_input_3', {'out': 0}),
                                    ('offset_in_input_3', 'lda', {'out': 0}),
                                    ('lda', 'offset_in_lda_1', {'out': 0}),
                                    ('input', 'offset_in_input_1', {'out': 1}),
                                    ('offset_in_lda_1', 'append_input_lda', {'in': 1, 'out': 0}),
                                    ('offset_in_input_1', 'append_input_lda', {'in': 0, 'out': 0}),
                                    ('append_input_lda', 'tdnn1.affine', {'out': 0}),
                                    ('tdnn1.affine', 'tdnn1.relu', {'out': 0}),
                                ]
                                )

        (flag, resp) = compare_graphs(graph, ref_graph, 'tdnn1.relu')
        self.assertTrue(flag, resp)

    def test_load_components(self):
        test_map = b"<NumComponents> " + struct.pack('B', 4) + struct.pack('I', 3) + \
                   b"<ComponentName> lda <FixedAffineComponent> <LinearParams> </FixedAffineComponent> " + \
                   b"<ComponentName> tdnn1.affine <NaturalGradientAffineComponent> <MaxChange>  @?<LearningRate> <LinearParams> </NaturalGradientAffineComponent> " + \
                   b"<ComponentName> tdnn1.relu <RectifiedLinearComponent> <ValueAvg> FV </RectifiedLinearComponent>"

        graph = build_graph({'input': {'shape': np.array([1, 16]), 'kind': 'op', 'op': 'Parameter'},
                             'lda': {'kind': 'op', 'op': 'fixedaffinecomponent'},
                             'tdnn1.affine': {'kind': 'op', 'op': 'fixedaffinecomponent'},
                             'tdnn1.relu': {'kind': 'op', 'op': 'relu'},
                             },
                            [
                                ('input', 'lda'),
                                ('lda', 'tdnn1.affine'),
                                ('tdnn1.affine', 'tdnn1.relu'),
                            ]
                            )

        ref_map = {b"lda": ["lda"],
                   b"tdnn1.affine": ["tdnn1.affine"],
                   b"tdnn1.relu": ["tdnn1.relu"]}

        load_components(io.BytesIO(test_map), graph, ref_map)

        ref_graph = build_graph({'input': {'shape': np.array([1, 16]), 'kind': 'op', 'op': 'Parameter'},
                                 'lda': {'kind': 'op', 'op': 'fixedaffinecomponent', 'parameters': '<LinearParams> '},
                                 'tdnn1.affine': {'kind': 'op', 'op': 'naturalgradientaffinecomponent', 'parameters': "<MaxChange>  @?<LearningRate> ·С8<LinearParams> "},
                                 'tdnn1.relu': {'kind': 'op', 'op': 'rectifiedlinearcomponent', 'parameters': "<Dim>   <ValueAvg> FV "},
                                 },
                                [
                                    ('input', 'lda'),
                                    ('lda', 'tdnn1.affine'),
                                    ('tdnn1.affine', 'tdnn1.relu'),
                                ]
                                )
        (flag, resp) = compare_graphs(graph, ref_graph, 'tdnn1.relu')
        self.assertTrue(flag, resp)

    def test_component_map_loading_scale(self):
        test_map = "input-node name=input dim=16\n" + \
                   "component-node name=lda component=lda input=Scale(0.1, input)\n" + \
                   "\n"
        graph = Graph(name="test_graph_component_map_loading_scale")

        test_top_map = load_topology_map(io.BytesIO(bytes(test_map, 'ascii')), graph)

        ref_map = {b"lda": ["lda"]}
        self.assertEqual(test_top_map, ref_map)
        self.assertTrue("input" in graph.nodes())
        self.assertListEqual(list(Node(graph, 'input')['shape']), [1, 16])

        ref_graph = build_graph({'input': {'shape': np.array([1, 16]), 'kind': 'op', 'op': 'Parameter'},
                                 'lda': {'kind': 'op'},
                                 'mul': {'kind': 'op'},
                                 'scale_const': {'kind': 'op', 'op': 'Const'},
                                 },
                                [
                                    ('input', 'mul', {'in': 0}),
                                    ('scale_const', 'mul', {'in': 1}),
                                    ('mul', 'lda', {'out': 0}),
                                ]
                                )

        (flag, resp) = compare_graphs(graph, ref_graph, 'lda')
        self.assertTrue(flag, resp)
