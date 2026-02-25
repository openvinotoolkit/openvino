# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import io
import os
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from openvino.tools.ovc.main import main
from openvino.tools.ovc.get_ov_update_message import get_compression_message


def arg_parse_helper(input_model,
                     use_legacy_frontend,
                     use_new_frontend,
                     input_model_is_text,
                     framework,
                     compress_to_fp16=False):
    path = os.path.dirname(__file__)
    input_model = os.path.join(path, "test_models", input_model)

    return argparse.Namespace(
        input_model=input_model,
        log_level='INFO',
        verbose=False,
        output_model=None,
        transform=[],
        output=None,
        input=None,
        compress_to_fp16=compress_to_fp16,
        extension=None
    )


class TestInfoMessagesTFFE(unittest.TestCase):
    @patch('argparse.ArgumentParser.parse_args',
           return_value=arg_parse_helper(input_model="model_int32.pbtxt",
                                         use_legacy_frontend=False, use_new_frontend=True,
                                         framework=None, input_model_is_text=True))
    @patch('openvino.tools.ovc.convert_impl.driver', side_effect=Exception('MESSAGE'))
    def run_fail_tf_fe(self, mock_driver):
        from openvino.tools.ovc import convert_model
        path = os.path.dirname(__file__)
        convert_model(os.path.join(path, "test_models", "model_int32.pbtxt"), verbose=True)


class TestInfoMessagesCompressFP16(unittest.TestCase):
    @patch('argparse.ArgumentParser.parse_args',
           return_value=arg_parse_helper(input_model="model_int32.pbtxt",
                                         use_legacy_frontend=False, use_new_frontend=True,
                                         compress_to_fp16=True,
                                         framework=None, input_model_is_text=True))
    def test_compress_to_fp16(self, mock_argparse):
        f = io.StringIO()
        with redirect_stdout(f):
            main()
            std_out = f.getvalue()
        fp16_compression_message_found = get_compression_message() in std_out
        assert fp16_compression_message_found
