# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import io
import re
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import numpy as np

import pytest


mock_available = True
try:
    # pylint: disable=no-name-in-module,import-error
    from mo.main import main

    # pylint: disable=no-name-in-module,import-error
    from mock_mo_python_api import get_frontend_statistic, get_model_statistic, \
        clear_frontend_statistic, clear_model_statistic, clear_place_statistic, \
        mock_return_partial_shape

    # pylint: disable=no-name-in-module,import-error
    from ngraph import PartialShape
    from ngraph.frontend import FrontEndManager
    from ngraph.utils.types import get_element_type

except Exception:
    print("No mock frontend API available,"
          "ensure to use -DENABLE_TESTS=ON option when running these tests")
    mock_available = False

# FrontEndManager shall be initialized and destroyed after all tests finished
# This is because destroy of FrontEndManager will unload all plugins,
# no objects shall exist after this
if mock_available:
    fem = FrontEndManager()

mock_needed = pytest.mark.skipif(not mock_available,
                                 reason="mock MO fe is not available")


def replaceArgsHelper(log_level='DEBUG',
                      silent=False,
                      model_name='abc',
                      input_model='abc.abc',
                      transform=[],
                      legacy_ir_generation=False,
                      scale=None,
                      output=None,
                      _input=None,
                      input_shape=None,
                      batch=None,
                      mean_values=None,
                      scale_values=None,
                      output_dir='.',
                      freeze_placeholder_with_value=None):
    return argparse.Namespace(
        log_level=log_level,
        silent=silent,
        model_name=model_name,
        input_model=input_model,
        transform=transform,
        legacy_ir_generation=legacy_ir_generation,
        scale=scale,
        output=output,
        input=_input,
        input_shape=input_shape,
        batch=batch,
        mean_values=mean_values,
        scale_values=scale_values,
        output_dir=output_dir,
        freeze_placeholder_with_value=freeze_placeholder_with_value)


class TestMainFrontend(unittest.TestCase):
    def setUp(self):
        clear_frontend_statistic()
        clear_model_statistic()
        clear_place_statistic()

    @mock_needed
    @patch('argparse.ArgumentParser.parse_args',
           return_value=replaceArgsHelper())
    def test_simple_convert(self, mock_argparse):
        f = io.StringIO()
        with redirect_stdout(f):
            main(argparse.ArgumentParser(), fem, 'mock_mo_ngraph_frontend')
            out = f.getvalue()

        xml_file = re.search(r'\[ SUCCESS \] XML file: (.*)', out).\
            group(1).replace("\r", "")
        bin_file = re.search(r'\[ SUCCESS \] BIN file: (.*)', out).\
            group(1).replace("\r", "")
        assert xml_file and bin_file

        # verify that 'convert' was called
        stat = get_frontend_statistic()
        assert stat.convert_model == 1
        # verify that meta info is added to XML file
        with open(xml_file) as file:
            assert 'mock_mo_ngraph_frontend' in file.read()

    @mock_needed
    @patch('argparse.ArgumentParser.parse_args',
           return_value=replaceArgsHelper(_input='newInput1,mock_input2'))
    def test_override_inputs(self, mock_argparse):

        main(argparse.ArgumentParser(), fem, 'mock_mo_ngraph_frontend')
        stat = get_model_statistic()

        # verify that 'override_all_inputs' was called
        assert stat.override_all_inputs == 1
        assert stat.override_all_outputs == 0
        assert stat.extract_subgraph == 0

    @mock_needed
    @patch('argparse.ArgumentParser.parse_args',
           return_value=replaceArgsHelper(output='newOut1,mock_output2'))
    def test_override_outputs(self, mock_argparse):
        main(argparse.ArgumentParser(), fem, 'mock_mo_ngraph_frontend')
        stat = get_model_statistic()

        # verify that 'override_all_outputs' was called
        assert stat.override_all_inputs == 0
        assert stat.override_all_outputs == 1
        assert stat.extract_subgraph == 0

    @mock_needed
    @patch('argparse.ArgumentParser.parse_args',
           return_value=replaceArgsHelper(_input='newIn1,newIn2',
                                          output='newOut1,newOut2'))
    def test_extract_subgraph(self, mock_argparse):
        main(argparse.ArgumentParser(), fem, 'mock_mo_ngraph_frontend')
        stat = get_model_statistic()

        # verify that 'extract_subgraph' was called
        assert stat.override_all_inputs == 0
        assert stat.override_all_outputs == 0
        assert stat.extract_subgraph == 1

    @mock_needed
    @patch('argparse.ArgumentParser.parse_args',
           return_value=replaceArgsHelper(_input='mock_input2,mock_input1',
                                          output='new_output2,mock_output1'))
    def test_override_same_inputs(self, mock_argparse):

        main(argparse.ArgumentParser(), fem, 'mock_mo_ngraph_frontend')
        stat = get_model_statistic()

        # verify that 'override_all_outputs' was called
        # because inputs were not changed
        assert stat.override_all_inputs == 0
        assert stat.override_all_outputs == 1
        assert stat.extract_subgraph == 0

    @mock_needed
    @patch('argparse.ArgumentParser.parse_args',
           return_value=replaceArgsHelper(_input='newInput1,mock_input2',
                                          output='mock_output2,mock_output1'))
    def test_override_same_outputs(self, mock_argparse):

        main(argparse.ArgumentParser(), fem, 'mock_mo_ngraph_frontend')
        stat = get_model_statistic()

        # verify that 'override_all_inputs' was called
        # because outputs were not changed
        assert stat.override_all_inputs == 1
        assert stat.override_all_outputs == 0
        assert stat.extract_subgraph == 0

    @mock_needed
    @patch('argparse.ArgumentParser.parse_args',
           return_value=replaceArgsHelper(_input='newIn1',
                                          input_shape='[1,2,3,4]'))
    def test_input_shape(self, mock_argparse):
        main(argparse.ArgumentParser(), fem, 'mock_mo_ngraph_frontend')
        stat = get_model_statistic()

        # verify that 'set_partial_shape' was called
        assert stat.set_partial_shape == 1
        assert stat.lastArgPartialShape == PartialShape([1, 2, 3, 4])

    @mock_needed
    @patch('argparse.ArgumentParser.parse_args',
           return_value=replaceArgsHelper(_input='newIn1{i8}'))
    def test_element_type(self, mock_argparse):
        main(argparse.ArgumentParser(), fem, 'mock_mo_ngraph_frontend')
        stat = get_model_statistic()

        # verify that 'set_element_type' was called
        assert stat.set_element_type == 1
        assert stat.lastArgElementType == get_element_type(np.int8)

    @mock_needed
    @patch('argparse.ArgumentParser.parse_args',
           return_value=replaceArgsHelper(batch=123))
    def test_set_batch_size(self, mock_argparse):
        mock_return_partial_shape(PartialShape([-1, 2, 3, 4]))
        main(argparse.ArgumentParser(), fem, 'mock_mo_ngraph_frontend')
        stat = get_model_statistic()

        # verify that 'set_element_type' was called
        # 2 is because mock model has 2 inputs
        assert stat.get_partial_shape == 2
        assert stat.set_partial_shape == 2
        assert stat.lastArgPartialShape == PartialShape([123, 2, 3, 4])

    @mock_needed
    @patch('argparse.ArgumentParser.parse_args',
           return_value=replaceArgsHelper(batch=123))
    def test_error_batch(self, mock_argparse):
        # First dimension doesn't look like a batch,
        # so MO shall not convert anything and produce specified error
        mock_return_partial_shape(PartialShape([122, 2, 3, 4]))
        with self.assertLogs() as logger:
            main(argparse.ArgumentParser(), fem, 'mock_mo_ngraph_frontend')

        stat = get_model_statistic()

        assert [s for s in logger.output if 'question=39' in s]

        # verify that 'get_element_type' was called
        assert stat.get_partial_shape == 1
        # verify that 'set_element_type' was not called
        assert stat.set_partial_shape == 0
