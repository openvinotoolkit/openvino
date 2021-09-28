# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import os
import tempfile
import unittest
from argparse import Namespace

from defusedxml.common import EntitiesForbidden

from mo.utils.ir_reader.restore_graph import restore_graph_from_ir, define_data_type
from unit_tests.utils.graph import build_graph


class TestIRReader(unittest.TestCase):
    def setUp(self):
        self.xml_bomb = b'<?xml version="1.0"?>\n' \
                   b'<!DOCTYPE lolz [\n' \
                   b' <!ENTITY lol "lol">\n' \
                   b' <!ELEMENT lolz (#PCDATA)>\n' \
                   b' <!ENTITY lol1 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;">\n' \
                   b' <!ENTITY lol2 "&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;">\n' \
                   b' <!ENTITY lol3 "&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;">\n' \
                   b' <!ENTITY lol4 "&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;">\n' \
                   b' <!ENTITY lol5 "&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;&lol4;">\n' \
                   b' <!ENTITY lol6 "&lol5;&lol5;&lol5;&lol5;&lol5;&lol5;&lol5;&lol5;&lol5;&lol5;">\n' \
                   b' <!ENTITY lol7 "&lol6;&lol6;&lol6;&lol6;&lol6;&lol6;&lol6;&lol6;&lol6;&lol6;">\n' \
                   b' <!ENTITY lol8 "&lol7;&lol7;&lol7;&lol7;&lol7;&lol7;&lol7;&lol7;&lol7;&lol7;">\n' \
                   b' <!ENTITY lol9 "&lol8;&lol8;&lol8;&lol8;&lol8;&lol8;&lol8;&lol8;&lol8;&lol8;">\n' \
                   b']>\n' \
                   b'<lolz>&lol9;</lolz>'

    def test_read_xml_bomb(self):
        bomb_file = tempfile.NamedTemporaryFile(delete=False)
        bomb_file.write(self.xml_bomb)
        bomb_file.close()
        self.assertRaises(EntitiesForbidden, restore_graph_from_ir, bomb_file.name)
        os.remove(bomb_file.name)


class TestDefineDataType(unittest.TestCase):
    nodes_attributes = {
        'input': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter', 'data_type': None},
        'const_1': {'kind': 'op', 'op': 'Const', 'data_type': None},
        'const_2': {'kind': 'op', 'op': 'Const', 'data_type': None},
        'operation_1': {'type': 'fake_op', 'kind': 'op', 'op': 'fake_op'},
        'operation_2': {'type': 'fake_op', 'kind': 'op', 'op': 'fake_op'},
        'output': {'type': None, 'value': None, 'kind': 'op', 'op': 'Result'},
    }

    def test_fp_16(self):
        graph = build_graph(self.nodes_attributes,
                            [
                                ('input', 'operation_1', {'in': 0}),
                                ('const_1', 'operation_1', {'in': 1}),
                                ('operation_1', 'operation_2', {'in': 0}),
                                ('const_2', 'operation_2', {'in': 1}),
                                ('operation_2', 'output')
                            ], {'const_1': {'data_type': np.float16},
                                'const_2': {'data_type': np.float32}},
                            nodes_with_edges_only=True, cli=Namespace(static_shape=False, data_type='FP16'))
        data_type = define_data_type(graph)

        self.assertEqual(data_type, 'FP16')

    def test_fp_32(self):
        graph = build_graph(self.nodes_attributes,
                            [
                                ('input', 'operation_1', {'in': 0}),
                                ('const_1', 'operation_1', {'in': 1}),
                                ('operation_1', 'operation_2', {'in': 0}),
                                ('const_2', 'operation_2', {'in': 1}),
                                ('operation_2', 'output')
                            ], {'const_1': {'data_type': np.float32},
                                'const_2': {'data_type': np.float32}},
                            nodes_with_edges_only=True, cli=Namespace(static_shape=False, data_type='FP32'))
        data_type = define_data_type(graph)

        self.assertEqual(data_type, 'FP32')

    def test_no_const_fp_16(self):
        graph = build_graph(self.nodes_attributes,
                            [
                                ('input', 'operation_1', {'in': 0}),
                                ('operation_1', 'operation_2', {'in': 0}),
                                ('operation_2', 'output')
                            ], {'input': {'data_type': np.float16}},
                            nodes_with_edges_only=True, cli=Namespace(static_shape=False, data_type='FP16'))
        data_type = define_data_type(graph)

        self.assertEqual(data_type, 'FP16')

    def test_no_const_fp_32(self):
        graph = build_graph(self.nodes_attributes,
                            [
                                ('input', 'operation_1', {'in': 0}),
                                ('operation_1', 'operation_2', {'in': 0}),
                                ('operation_2', 'output')
                            ], {'input': {'data_type': np.float32}},
                            nodes_with_edges_only=True, cli=Namespace(static_shape=False, data_type='FP32'))
        data_type = define_data_type(graph)

        self.assertEqual(data_type, 'FP32')
