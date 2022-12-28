# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import io
import os
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from openvino.tools.mo.main import main
from openvino.tools.mo.utils.get_ov_update_message import get_tf_fe_message, get_tf_fe_legacy_message


def arg_parse_helper(input_model,
                     use_legacy_frontend,
                     use_new_frontend,
                     input_model_is_text,
                     framework):
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
        freeze_placeholder_with_value=None,
        tensorflow_use_custom_operations_config=None,
        data_type=None,
        tensorflow_custom_operations_config_update=None,
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
        tf_fe_legacy_message_found = get_tf_fe_legacy_message() in std_out
        assert tf_fe_message_found and not tf_fe_legacy_message_found

    @patch('argparse.ArgumentParser.parse_args',
           return_value=arg_parse_helper(input_model="future_op.pbtxt",
                                         use_legacy_frontend=True, use_new_frontend=False,
                                         framework=None, input_model_is_text=True))
    def test_tf_fe_legacy(self, mock_argparse):
        f = io.StringIO()
        with redirect_stdout(f):
            main(argparse.ArgumentParser())
            std_out = f.getvalue()
        tf_fe_message_found = get_tf_fe_message() in std_out
        tf_fe_legacy_message_found = get_tf_fe_legacy_message() in std_out
        assert tf_fe_legacy_message_found and not tf_fe_message_found
