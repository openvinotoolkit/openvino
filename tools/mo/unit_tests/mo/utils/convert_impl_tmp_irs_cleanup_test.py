# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import unittest
from unittest.mock import patch

from openvino.tools.mo.convert import convert_model
from openvino.tools.mo.utils.error import Error
from unit_tests.utils.utils import base_args_config


class TestConvertImplTmpIrsCleanup(unittest.TestCase):
    test_model_file = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "moc_tf_fe/test_models/mul_with_unknown_rank_y.pbtxt") 

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
            emit_ir_func.side_effect = Exception("Error during FEM read_model")

            args = base_args_config()
            args.input_model = self.test_model_file
            args.input = "x[3],y[1 3]"
            args.input_model_is_text = True

            self.assertRaises(Exception, convert_model, **vars(args))
            self.assertFalse(self.are_tmp_files_left(self.test_model_file))
