# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log
import numpy as np
import os
import sys
import unittest
from openvino.tools.mo.front.common.partial_infer.utils import shape_array, strict_compare_tensors
from openvino.tools.mo.graph.graph import Node
from openvino.tools.mo.utils.ir_engine.ir_engine import IREngine
from unittest import mock

log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.DEBUG, stream=sys.stdout)


class TestFunction(unittest.TestCase):
    def setUp(self):
        path, _ = os.path.split(os.path.dirname(__file__))
        self.xml = os.path.join(path, os.pardir, os.pardir,
                                "utils", "test_data", "mxnet_synthetic_gru_bidirectional_FP16_1_v6.xml")
        self.xml_negative = os.path.join(path, os.pardir, os.pardir,
                                         "utils", "test_data",
                                         "mxnet_synthetic_gru_bidirectional_FP16_1_v6_negative.xml")
        self.bin = os.path.splitext(self.xml)[0] + '.bin'
        self.assertTrue(os.path.exists(self.xml), 'XML file not found: {}'.format(self.xml))
        self.assertTrue(os.path.exists(self.bin), 'BIN file not found: {}'.format(self.bin))

        self.IR = IREngine(path_to_xml=str(self.xml), path_to_bin=str(self.bin))
        self.IR_ref = IREngine(path_to_xml=str(self.xml), path_to_bin=str(self.bin))
        self.IR_negative = IREngine(path_to_xml=str(self.xml_negative), path_to_bin=str(self.bin))

    def test_is_float(self):
        test_cases = [(4.4, True), ('aaaa', False)]
        for test_data, result in test_cases:
            test_data = test_data
            self.assertEqual(IREngine._IREngine__isfloat(test_data), result,
                             "Function __isfloat is not working with value: {}".format(test_data))
            log.info(
                'Test for function __is_float passed with value: {}, expected result: {}'.format(test_data, result))

    # TODO add comparison not for type IREngine
    def test_compare(self):
        flag, msg = self.IR.compare(self.IR_ref)
        self.assertTrue(flag, 'Comparing false, test compare function failed')
        log.info('Test for function compare passed')

    def test_compare_negative(self):
        # Reference data for test:
        reference_msg = 'Current node "2" with type "Const" and reference node "2" with type "Input" have different ' \
                        'attr "type" : Const and Input'
        # Check function:
        flag, msg = self.IR.compare(self.IR_negative)
        self.assertFalse(flag, 'Comparing flag failed, test compare function failed')
        self.assertEqual('\n'.join(msg), reference_msg, 'Comparing message failed, test compare negative failed')

        log.info('Test for function compare passed')

    def test_find_input(self):
        # Create references for this test:
        ref_nodes = [Node(self.IR.graph, '0')]
        # Check function:
        a = IREngine._IREngine__find_input(self.IR.graph)
        self.assertTrue(a == ref_nodes, 'Error')

    def test_get_inputs(self):
        # Reference data for test:
        ref_input_dict = {'data': shape_array([1, 10, 16])}
        # Check function:
        inputs_dict = self.IR.get_inputs()
        self.assertTrue(strict_compare_tensors(ref_input_dict['data'], inputs_dict['data']),
                        'Test on function get_inputs failed')
        log.info('Test for function get_inputs passed')

    def test_eq_function(self):
        self.assertTrue(self.IR == self.IR_ref, 'Comparing false, test eq function failed')
        log.info('Test for function eq passed')

    @unittest.mock.patch('numpy.savez_compressed')
    def test_generate_bin_hashes_file(self, numpy_savez):
        # Generate bin_hashes file in default directory
        self.IR.generate_bin_hashes_file()
        numpy_savez.assert_called_once()
        log.info('Test for function generate_bin_hashes_file with default folder passed')

    @unittest.mock.patch('numpy.savez_compressed')
    def test_generate_bin_hashes_file_custom_directory(self, numpy_savez):
        # Generate bin_hashes file in custom directory
        directory_for_file = os.path.join(os.path.split(os.path.dirname(__file__))[0], "utils", "test_data",
                                          "bin_hash")
        self.IR.generate_bin_hashes_file(path_for_file=directory_for_file)
        numpy_savez.assert_called_once()
        log.info('Test for function generate_bin_hashes_file with custom folder passed')

    def test_normalize_attr(self):
        test_cases = [({'order': '1,0,2'}, {'order': [1, 0, 2]}),
                      ({'order': '1'}, {'order': 1})]
        for test_data, reference in test_cases:
            result_dict = IREngine._IREngine__normalize_attrs(attrs=test_data)
            self.assertTrue(reference == result_dict, 'Test on function normalize_attr failed')
            log.info('Test for function normalize_attr passed')

    def test_load_bin_hashes(self):
        path_for_file = self.IR.generate_bin_hashes_file()
        IR = IREngine(path_to_xml=str(self.xml), path_to_bin=str(path_for_file))
        is_ok = True
        # Check for constant nodes
        const_nodes = IR.graph.get_op_nodes(type='Const')
        for node in const_nodes:
            if not node.has_valid('hashes'):
                log.error('Constant node {} do not include hashes'.format(node.name))
                is_ok = False

        # Check for TensorIterator Body
        ti_nodes = IR.graph.get_op_nodes(type='TensorIterator')
        for ti in ti_nodes:
            if not ti.has_valid('body'):
                log.error("TensorIterator doesn't have body attribute for node: {}".format(ti.name))
            else:
                const_ti_nodes = ti.body.graph.get_op_nodes(type='Const')
                for node in const_ti_nodes:
                    if not node.has_valid('hashes'):
                        log.error('Constant node {} do not include hashes'.format(node.name))
                        is_ok = False

        self.assertTrue(is_ok, 'Test for function load_bin_hashes failed')
        os.remove(path_for_file)

    def test_isint(self):
        test_cases = [
            ("0", True),
            ("1", True),
            ("-1", True),
            ("-", False),
            ("+1", True),
            ("+", False),
            ("1.0", False),
            ("-1.0", False),
            ("1.5", False),
            ("+1.5", False),
            ("abracadabra", False)]
        for value, result in test_cases:
            self.assertEqual(IREngine._IREngine__isint(value), result)
