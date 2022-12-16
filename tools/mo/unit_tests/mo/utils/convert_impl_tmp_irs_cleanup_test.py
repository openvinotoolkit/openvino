# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import unittest
from unittest.mock import patch

from openvino.tools.mo.convert import convert_model
from openvino.tools.mo.utils.error import Error


def base_args_config():
    args = argparse.Namespace()
    args.extensions = [os.getcwd()]
    args.use_legacy_frontend = False
    args.use_new_frontend = False
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
    args.mean_values = ()
    args.scale_values = ()
    args.output_dir = os.getcwd()
    args.freeze_placeholder_with_value = None
    args.transformations_config = None
    args.disable_fusing = None
    args.finegrain_fusing = None
    args.disable_resnet_optimization = None
    args.enable_concat_optimization = None
    args.static_shape = None
    args.disable_weights_compression = None
    args.reverse_input_channels = None
    args.data_type = None
    args.layout = ()
    args.source_layout = ()
    args.target_layout = ()
    args.input_checkpoint = None
    args.saved_model_dir = None
    args.input_meta_graph = None
    args.saved_model_tags = None
    args.progress = True
    args.stream_output = False
    args.tensorflow_use_custom_operations_config = None
    args.tensorflow_custom_layer_libraries = None
    args.tensorflow_custom_operations_config_update = None
    args.tensorboard_logdir = None
    args.disable_nhwc_to_nchw = False
    return args


class TestConvertImplTmpIrsCleanup(unittest.TestCase):
    test_model_file = "../../moc_tf_fe/test_models/mul_with_unknown_rank_y.pbtxt"

    @staticmethod
    def are_tmp_files_left(orig_model_name):
        for suf in [".xml", ".bin", ".mapping"]:
            path_to_file = orig_model_name.replace('.pbtxt', '_tmp' + suf)
            if os.path.exists(path_to_file):
                return True
        return False

    def test_tmp_irs_cleanup_convert_impl_1(self):
        with patch("openvino.tools.mo.back.offline_transformations.apply_offline_transformations") as emit_ir_func:
            emit_ir_func.side_effect = Error("Error during offline_transformation")

            args = base_args_config()
            args.input_model = self.test_model_file
            args.input = "x[3],y[1 3]"
            args.input_model_is_text = True

            self.assertRaises(Error, convert_model, **vars(args))
            self.assertFalse(self.are_tmp_files_left(self.test_model_file))

    def test_tmp_irs_cleanup_convert_impl_2(self):
        with patch("openvino.tools.mo.back.ie_ir_ver_2.emitter.add_net_rt_info") as emit_ir_func:
            emit_ir_func.side_effect = Error("Error during tmp emitting")

            args = base_args_config()
            args.input_model = self.test_model_file
            args.input = "x[3],y[1 3]"
            args.input_model_is_text = True

            self.assertRaises(Error, convert_model, **vars(args))
            self.assertFalse(self.are_tmp_files_left(self.test_model_file))


    def test_tmp_irs_cleanup_convert_impl_3(self):
        with patch("openvino.tools.mo.convert_impl.read_model") as emit_ir_func:
            emit_ir_func.side_effect = Error("Error during FEM read_model")

            args = base_args_config()
            args.input_model = self.test_model_file
            args.input = "x[3],y[1 3]"
            args.input_model_is_text = True

            self.assertRaises(Error, convert_model, **vars(args))
            self.assertFalse(self.are_tmp_files_left(self.test_model_file))
