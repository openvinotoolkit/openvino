# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import tempfile
import unittest

from openvino.tools.mo.front.tf.loader import convert_to_pb


class ConvertToPBTests(unittest.TestCase):
    test_directory = os.path.dirname(os.path.realpath(__file__))

    def setUp(self):
        self.argv = argparse.Namespace(input_model=None, input_model_is_text=False, input_checkpoint=None, output=None,
                                       saved_model_dir=None, input_meta_graph=None, saved_model_tags=None,
                                       model_name='model', output_dir=None)

    def test_saved_model(self):
        import tensorflow as tf
        with tempfile.TemporaryDirectory(dir=self.test_directory) as tmp_dir:
            inputs = tf.keras.Input(shape=(3,))
            x = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs)
            outputs = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(x)
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            model.save(tmp_dir)
            self.argv.saved_model_dir = tmp_dir
            self.argv.output_dir = tmp_dir
            path_to_pb = convert_to_pb(self.argv)
            self.assertTrue(os.path.exists(path_to_pb), "The auxiliary .pb is not generated")
            self.assertTrue(os.path.getsize(path_to_pb) != 0, "The auxiliary .pb is empty")

    def test_meta_format(self):
        try:
            import tensorflow.compat.v1 as tf_v1
            tf_v1.disable_eager_execution()
        except ImportError:
            import tensorflow as tf_v1

        with tempfile.TemporaryDirectory(dir=self.test_directory) as tmp_dir:
            a = tf_v1.get_variable("A", initializer=tf_v1.constant(3, shape=[2]))
            b = tf_v1.get_variable("B", initializer=tf_v1.constant(5, shape=[2]))
            tf_v1.add(a, b, name='Add')
            init_op = tf_v1.global_variables_initializer()
            saver = tf_v1.train.Saver()
            with tf_v1.Session() as sess:
                sess.run(init_op)
                saver.save(sess, os.path.join(tmp_dir, 'model'))
            self.argv.input_meta_graph = os.path.join(tmp_dir, 'model.meta')
            self.argv.output_dir = tmp_dir
            path_to_pb = convert_to_pb(self.argv)
            self.assertTrue(os.path.exists(path_to_pb), "The auxiliary .pb is not generated")
            self.assertTrue(os.path.getsize(path_to_pb) != 0, "The auxiliary .pb is empty")
