# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
import unittest
from pathlib import Path

from openvino import get_version as get_rt_version
from openvino import serialize, convert_model
from openvino.tools.mo.utils.ir_reader.restore_graph import restore_graph_from_ir, save_restored_graph


class MetaDataTestTF(unittest.TestCase):
    test_directory = os.path.dirname(os.path.realpath(__file__))

    @staticmethod
    def check_meta_data(ov_model, ref_meta):
        ignore_attrs = ['version', 'optimization']
        for key, value in ref_meta.items():
            if key == 'conversion_parameters':
                for param_name, param_value in value.items():
                    val = ov_model.get_rt_info([key, param_name]).astype(str)
                    if param_name in ['extension',  'input_model']:
                        val = Path(val)
                    assert val == param_value, \
                        "Runtime info attribute with name {} does not match. Expected: {}, " \
                        "got {}".format(param_name, param_value, val)
                continue
            assert ov_model.get_rt_info(key).astype(str) == value, \
                "Runtime info attribute with name {} does not match. Expected: {}, " \
                "got {}".format(key, value, ov_model.get_rt_info(key).astype(str))

        for key, value in ov_model.get_rt_info().items():
            if key in ignore_attrs:
                continue
            assert key in ref_meta, "Unexpected runtime info attribute: {}".format(key)

    def test_meta_data_tf(self):
        def create_tf_model(out_dir):
            import tensorflow as tf

            tf.compat.v1.reset_default_graph()

            with tf.compat.v1.Session() as sess:
                inp1 = tf.compat.v1.placeholder(tf.float32, [1, 2, 3], 'Input')
                inp2 = tf.compat.v1.placeholder(tf.float32, [1, 2, 3], 'Input')
                relu = tf.nn.relu(inp1 + inp2, name='Relu')

                output = tf.nn.sigmoid(relu, name='Sigmoid')

                tf.compat.v1.global_variables_initializer()
                tf_net = sess.graph_def
            tf.io.write_graph(tf_net, out_dir + os.sep, 'model_bool.pb', as_text=False)
            return out_dir + os.sep + 'model_bool.pb'

        def ref_meta_data():
            return {
                'Runtime_version': get_rt_version(),
                'legacy_frontend': "False",
                'conversion_parameters': {
                    'scale': "1.5",
                    'batch': "1"
                }
            }

        with tempfile.TemporaryDirectory(dir=self.test_directory) as tmpdir:
            model = create_tf_model(tmpdir)
            out_xml = os.path.join(tmpdir, "model.xml")
            ref_meta = ref_meta_data()

            ov_model = convert_model(model, scale=1.5, batch=1)
            self.check_meta_data(ov_model, ref_meta)

            serialize(ov_model, out_xml.encode('utf-8'), out_xml.replace('.xml', '.bin').encode('utf-8'))

            from openvino import Core
            core = Core()
            deserialized_model = core.read_model(out_xml)
            self.check_meta_data(deserialized_model, ref_meta)

            restored_graph, meta_data = restore_graph_from_ir(out_xml, out_xml.replace('.xml', '.bin'))
            save_restored_graph(restored_graph, tmpdir, meta_data, "mo_ir_reader_test_model")

            mo_ir_reader_test_model = core.read_model(os.path.join(tmpdir, "mo_ir_reader_test_model.xml"))
            self.check_meta_data(mo_ir_reader_test_model, ref_meta)
