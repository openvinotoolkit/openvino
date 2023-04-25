# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import io
import os
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from openvino.tools.mo.main import main
from openvino.tools.mo.subprocess_main import setup_env
from openvino.tools.mo.utils.get_ov_update_message import get_tf_fe_message, get_compression_message


def arg_parse_helper(input_model,
                     use_legacy_frontend,
                     use_new_frontend,
                     input_model_is_text,
                     framework,
                     compress_to_fp16=False,
                     freeze_placeholder_with_value=None):
    path = os.path.dirname(__file__)
    input_model = os.path.join(path, "test_models", input_model)

    return argparse.Namespace(
        input_model=input_model,
        use_legacy_frontend=use_legacy_frontend,
        use_new_frontend=use_new_frontend,
        framework=framework,
        input_model_is_text=input_model_is_text,
        log_level='INFO',
        silent=True,
        model_name=None,
        transform=[],
        scale=None,
        output=None,
        input=None,
        input_shape=None,
        batch=None,
        input_checkpoint=None,
        saved_model_dir=None,
        input_meta_graph=None,
        saved_model_tags=None,
        output_dir='.',
        mean_values=(),
        scale_values=(),
        layout={},
        source_layout={},
        target_layout={},
        freeze_placeholder_with_value=freeze_placeholder_with_value,
        data_type=None,
        tensorflow_custom_operations_config_update=None,
        compress_to_fp16=compress_to_fp16,
        extensions=None
    )


class TestInfoMessagesTFFE(unittest.TestCase):
    @patch('argparse.ArgumentParser.parse_args',
           return_value=arg_parse_helper(input_model="model_int32.pbtxt",
                                         use_legacy_frontend=False, use_new_frontend=True,
                                         framework=None, input_model_is_text=True))
    def test_api20_only(self, mock_argparse):
        f = io.StringIO()
        with redirect_stdout(f):
            main(argparse.ArgumentParser())
            std_out = f.getvalue()
        tf_fe_message_found = get_tf_fe_message() in std_out
        assert tf_fe_message_found


class TestInfoMessagesTFFEWithFallback(unittest.TestCase):
    @patch('argparse.ArgumentParser.parse_args',
           return_value=arg_parse_helper(input_model="model_switch_merge.pbtxt",
                                         use_legacy_frontend=False, use_new_frontend=False,
                                         framework=None, input_model_is_text=True,
                                         freeze_placeholder_with_value="is_training->False"))
    def test_tf_fe_message_fallback(self, mock_argparse):
        f = io.StringIO()
        with redirect_stdout(f):
            main(argparse.ArgumentParser())
            std_out = f.getvalue()
        tf_fe_message_found = get_tf_fe_message() in std_out
        assert not tf_fe_message_found, 'TF FE Info message is found for the fallback case'


class TestInfoMessagesCompressFP16(unittest.TestCase):
    @patch('argparse.ArgumentParser.parse_args',
           return_value=arg_parse_helper(input_model="model_int32.pbtxt",
                                         use_legacy_frontend=False, use_new_frontend=True,
                                         compress_to_fp16=True,
                                         framework=None, input_model_is_text=True))
    def test_compress_to_fp16(self, mock_argparse):
        f = io.StringIO()
        with redirect_stdout(f):
            main(argparse.ArgumentParser())
            std_out = f.getvalue()
        fp16_compression_message_found = get_compression_message() in std_out
        assert fp16_compression_message_found


class TestUnsatisfiedDependenciesMessages(unittest.TestCase):
    def test_mo_dependencies_check(self):
        setup_env()
        args = [sys.executable,
                os.path.join(os.path.dirname(__file__), '../mo/convert/dependencies_check_test1_actual.py')]

        # Check that not satisfied dependencies are shown if error happened in convert_model()
        status = subprocess.run(args, env=os.environ, capture_output=True)
        test_log = status.stderr.decode("utf-8").replace("\r\n", "\n")
        print(status.stderr.decode("utf-8").replace("\r\n", "\n"))
        print(status.stdout.decode("utf-8").replace("\r\n", "\n"))
        assert "Detected not satisfied dependencies:" in test_log
        assert "numpy: not installed" in test_log

        args = [sys.executable,
                os.path.join(os.path.dirname(__file__), '../mo/convert/dependencies_check_test2_actual.py')]

        # Check that not satisfied dependencies are not shown if convert_model() successfully converted the model
        status = subprocess.run(args, env=os.environ, capture_output=True)
        test_log = status.stdout.decode("utf-8").replace("\r\n", "\n")
        assert "Detected not satisfied dependencies:" not in test_log

    def test_mo_convert_logger(self):
        setup_env()
        args = [sys.executable,
                os.path.join(os.path.dirname(__file__), '../mo/convert/logger_test_actual.py')]

        status = subprocess.run(args, env=os.environ, capture_output=True)
        test_log = status.stdout.decode("utf-8").replace("\r\n", "\n")

        assert "test message 1" in test_log
        assert "test message 2" in test_log
        assert "test message 3" in test_log

        assert test_log.count("[ SUCCESS ] Total execution time") == 2

        # Check no error happens during importing modules from requirements
        assert 'Error happened while importing' not in test_log
        assert 'package error' not in test_log