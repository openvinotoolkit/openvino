# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import unittest
from unittest.mock import patch

from openvino.tools.mo.convert import convert_model
from openvino.tools.mo.utils.error import Error


class TestConvertImplTmpIrsCleanup(unittest.TestCase):
    test_model_file = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir,
                                   "moc_tf_fe/test_models/mul_with_unknown_rank_y.pbtxt")

    @staticmethod
    def are_tmp_files_left(orig_model_name):
        for suf in [".xml", ".bin", ".mapping"]:
            path_to_file = orig_model_name.replace('.pbtxt', '_tmp' + suf)
            if os.path.exists(path_to_file):
                return True
        return False

    def test_tmp_irs_cleanup_convert_impl_1(self):
        with patch("openvino.tools.mo.back.offline_transformations.apply_offline_transformations") as emit_ir_func:
            emit_ir_func.side_effect = Error('offline transformations step has failed')

            params = {'input_model': self.test_model_file, 'input_model_is_text': True, 'input': 'x[3],y[1 3]',
                      'use_legacy_frontend': True}
            self.assertRaisesRegex(Error, 'offline transformations step has failed', convert_model, **params)
            self.assertFalse(self.are_tmp_files_left(self.test_model_file))

    def test_tmp_irs_cleanup_convert_impl_2(self):
        with patch("openvino.tools.mo.back.ie_ir_ver_2.emitter.add_net_rt_info") as emit_ir_func:
            emit_ir_func.side_effect = Error('emitting tmp IR has failed')

            params = {'input_model': self.test_model_file, 'input_model_is_text': True, 'input': 'x[3],y[1 3]',
                      'use_legacy_frontend': True}
            self.assertRaisesRegex(Error, 'emitting tmp IR has failed', convert_model, **params)
            self.assertFalse(self.are_tmp_files_left(self.test_model_file))

    def test_tmp_irs_cleanup_convert_impl_3(self):
        with patch("openvino.tools.mo.convert_impl.read_model") as emit_ir_func:
            emit_ir_func.side_effect = Exception('FEM read_model has failed')

            params = {'input_model': self.test_model_file, 'input_model_is_text': True, 'input': 'x[3],y[1 3]',
                      'use_legacy_frontend': True}
            self.assertRaisesRegex(Error, 'FEM read_model has failed', convert_model, **params)
            self.assertFalse(self.are_tmp_files_left(self.test_model_file))
