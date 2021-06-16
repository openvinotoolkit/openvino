# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
from contextlib import redirect_stdout
import io
import os
import pytest
import sys
import unittest
from unittest.mock import patch

from mo.utils.cli_parser import get_absolute_path
from mo.main import main


mock_available = True
try:
    from ngraph import PartialShape
    from ngraph.frontend import FrontEndCapabilities, FrontEndManager, InitializationFailure
    from ngraph.utils.types import get_element_type
    from mock_mo_python_api import get_fe_stat, get_mdl_stat, get_place_stat
    print("Everything is available for Mock MO Frontend")
except Exception:
    print("No mock frontend API available, ensure to use -DENABLE_TESTS=ON option when running these tests")
    mock_available = False

# FrontEndManager shall be initialized and destroyed after all tests finished
# This is because destroy of FrontEndManager will unload all plugins, no objects shall exist after this
if mock_available:
    fem = FrontEndManager()

mock_needed = pytest.mark.skipif(not mock_available,
                                 reason="mock MO fe is not available")


def replaceArgsHelper(log_level='ERROR',
                      silent=False,
                      model_name='abc',
                      input_model='abc.bin',
                      transform=[],
                      legacy_ir_generation=False,
                      scale=None,
                      output=None,
                      input=None,
                      input_shape=None,
                      batch=None,
                      mean_values=None,
                      scale_values=None,
                      output_dir=get_absolute_path('.'),
                      freeze_placeholder_with_value=None):
    return argparse.Namespace(log_level=log_level,
                              silent=silent,
                              model_name=model_name,
                              input_model=input_model,
                              transform=transform,
                              legacy_ir_generation=legacy_ir_generation,
                              scale=scale,
                              output=output,
                              input=input,
                              input_shape=input_shape,
                              batch=batch,
                              mean_values=mean_values,
                              scale_values=scale_values,
                              output_dir=output_dir,
                              freeze_placeholder_with_value=freeze_placeholder_with_value)


class TestMainFrontend(unittest.TestCase):
    @mock_needed
    @patch('argparse.ArgumentParser.parse_args', return_value=replaceArgsHelper())
    def test_SimpleConvert(self, mock_argparse):
        statBefore = get_fe_stat()
        f = io.StringIO()
        with redirect_stdout(f):
            main(argparse.ArgumentParser(), fem, 'mock_mo_ngraph_frontend')
            out = f.getvalue()

        xml = '[ SUCCESS ] XML file' in out
        bin = '[ SUCCESS ] BIN file' in out
        assert xml and bin

        # verify that 'convert' was called
        statAfter = get_fe_stat()
        assert statAfter.convert_model == statBefore.convert_model + 1
