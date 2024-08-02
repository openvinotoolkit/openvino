# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import pytest
import unittest
from openvino.frontend import FrontEndManager, FrontEnd  # pylint: disable=no-name-in-module,import-error
from openvino.tools.mo.convert_impl import prepare_ir
from unittest.mock import Mock

try:
    import openvino_telemetry as tm
    from openvino_telemetry.backend import backend_ga4
except ImportError:
    import openvino.tools.mo.utils.telemetry_stub as tm


def base_args_config(use_legacy_fe: bool = None, use_new_fe: bool = None):
    args = argparse.Namespace()
    args.feManager = FrontEndManager()
    args.extensions = None
    args.use_legacy_frontend = use_legacy_fe
    args.use_new_frontend = use_new_fe
    args.framework = 'tf'
    args.model_name = None
    args.input_model = None
    args.silent = True
    args.transform = []
    args.scale = None
    args.output = None
    args.input = None
    args.input_shape = None
    args.batch = None
    args.mean_values = None
    args.scale_values = None
    args.output_dir = os.getcwd()
    args.freeze_placeholder_with_value = None
    args.transformations_config = None
    args.static_shape = None
    args.reverse_input_channels = None
    args.data_type = None
    args.layout = None
    args.source_layout = None
    args.target_layout = None
    args.input_model_is_text = False
    args.input_checkpoint = None
    args.saved_model_dir = None
    args.input_meta_graph = None
    args.saved_model_tags = None
    return args


class TestMoFallback(unittest.TestCase):
    test_directory = os.path.dirname(os.path.realpath(__file__))

    def create_fake_json_file(self, output_dir):
        json_data = '[]'  # json format
        json_path = os.path.join(output_dir, 'fake_config.json')
        with open(json_path, 'w') as f:
            f.write(json_data)
        return json_path

    def create_tensorflow_model_pb(self, output_dir):
        import tensorflow as tf
        try:
            import tensorflow.compat.v1 as tf_v1
        except ImportError:
            import tensorflow as tf_v1

        tf_v1.reset_default_graph()
        with tf_v1.Session() as sess:
            x = tf_v1.placeholder(tf.float32, [2, 3], 'x')
            y = tf_v1.placeholder(tf.float32, [2, 3], 'y')
            tf.add(x, y, name="add")
            tf_v1.global_variables_initializer()
            tf.io.write_graph(sess.graph, output_dir, 'model.pb', as_text=False)
        return os.path.join(output_dir, 'model.pb')

    def create_tensorflow_saved_model(self, output_dir):
        import tensorflow as tf
        inputs = tf.keras.Input(shape=(3,))
        x = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs)
        outputs = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model_path = os.path.join(output_dir, 'saved_model')
        model.save(model_path)
        return model_path

    def setUp(self):
        tm.Telemetry.__init__ = Mock(return_value=None)
        tm.Telemetry.send_event = Mock()
        FrontEnd.add_extension = Mock()

    def test_transform_config_fallback_tf_fe_pb(self):
        import tempfile
        test_cases = [
            # transformation config fallback even for use_new_frontend in case TF FE
            # TODO: uncomment this case once TF FE is unbricked and obtains normal name openvino_tensorflow_frontend
            # (False, True, 'mo_legacy', 'transformations_config'),
            # no fallback since legacy FE is used
            (True, False, 'mo_legacy', None),
            # no fallback since legacy FE is default for TensorFlow
            (False, False, 'mo_legacy', None)
        ]
        for use_legacy_frontend, use_new_frontend, expected_frontend, fallback_reason in test_cases:
            with tempfile.TemporaryDirectory(dir=self.test_directory) as tmp_dir:
                args = base_args_config(use_legacy_frontend, use_new_frontend)
                model_path = self.create_tensorflow_model_pb(tmp_dir)
                args.input_model = model_path
                args.framework = 'tf'
                args.transformations_config = self.create_fake_json_file(tmp_dir)

                prepare_ir(args)

                tm.Telemetry.send_event.assert_any_call('mo', 'conversion_method', expected_frontend)
                if fallback_reason:
                    tm.Telemetry.send_event.assert_any_call('mo', 'fallback_reason', fallback_reason)
                else:
                    with pytest.raises(AssertionError):  # not called
                        tm.Telemetry.send_event.assert_any_call('mo', 'fallback_reason', fallback_reason)

    def test_transform_config_fallback_tf_fe_saved_model(self):
        import tempfile
        test_cases = [
            # transformation config fallback even for use_new_frontend in case TF FE
            # TODO: uncomment this case once TF FE is unbricked and obtains normal name openvino_tensorflow_frontend
            # (False, True, 'mo_legacy', 'transformations_config'),
            # no fallback since legacy FE is used
            (True, False, 'mo_legacy', None),
            # no fallback since legacy FE is default for TensorFlow
            (False, False, 'mo_legacy', None),
        ]
        for use_legacy_frontend, use_new_frontend, expected_frontend, fallback_reason in test_cases:
            with tempfile.TemporaryDirectory(dir=self.test_directory) as tmp_dir:
                args = base_args_config(use_legacy_frontend, use_new_frontend)
                model_path = self.create_tensorflow_saved_model(tmp_dir)
                args.saved_model_dir = model_path
                args.framework = 'tf'
                args.transformations_config = self.create_fake_json_file(tmp_dir)

                print("args = ", args)
                prepare_ir(args)

                tm.Telemetry.send_event.assert_any_call('mo', 'conversion_method', expected_frontend)
                if fallback_reason:
                    tm.Telemetry.send_event.assert_any_call('mo', 'fallback_reason', fallback_reason)
                else:
                    with pytest.raises(AssertionError):  # not called
                        tm.Telemetry.send_event.assert_any_call('mo', 'fallback_reason', fallback_reason)
