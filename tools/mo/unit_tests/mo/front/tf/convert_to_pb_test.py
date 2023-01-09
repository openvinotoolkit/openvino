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
        except ImportError:
            import tensorflow as tf_v1
        from tensorflow.python.eager.context import graph_mode

        with tempfile.TemporaryDirectory(dir=self.test_directory) as tmp_dir:
            with graph_mode():
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

    def test_text_frozen_format(self):
        try:
            import tensorflow.compat.v1 as tf_v1
        except ImportError:
            import tensorflow as tf_v1
        tf_v1.reset_default_graph()

        # Create the graph and model
        with tempfile.TemporaryDirectory(dir=self.test_directory) as tmp_dir:
            with tf_v1.Session() as sess:
                x = tf_v1.placeholder(tf_v1.float32, [2, 3], 'x')
                y = tf_v1.placeholder(tf_v1.float32, [2, 3], 'y')
                tf_v1.add(x, y, name="add")

                tf_v1.global_variables_initializer()
                tf_v1.io.write_graph(sess.graph, tmp_dir, 'model.pbtxt', as_text=True)

            # initialize test case and check
            self.argv.input_model = os.path.join(tmp_dir, 'model.pbtxt')
            self.argv.input_model_is_text = True
            self.argv.output_dir = tmp_dir
            self.assertTrue(os.path.exists(self.argv.input_model),
                            "The test model in frozen text format must exist")
            # test convert_to_pb
            path_to_pb = convert_to_pb(self.argv)
            self.assertTrue(os.path.exists(path_to_pb), "The auxiliary .pb is not generated")
            self.assertTrue(os.path.getsize(path_to_pb) != 0, "The auxiliary .pb is empty")

    def test_binary_frozen_format(self):
        try:
            import tensorflow.compat.v1 as tf_v1
        except ImportError:
            import tensorflow as tf_v1
        tf_v1.reset_default_graph()

        # Create the graph and model
        with tempfile.TemporaryDirectory(dir=self.test_directory) as tmp_dir:
            with tf_v1.Session() as sess:
                x = tf_v1.placeholder(tf_v1.float32, [2, 3], 'x')
                y = tf_v1.placeholder(tf_v1.float32, [2, 3], 'y')
                tf_v1.add(x, y, name="add")

                tf_v1.global_variables_initializer()
                tf_v1.io.write_graph(sess.graph, tmp_dir, 'model.pb', as_text=False)

            # initialize test case and check
            self.argv.input_model = os.path.join(tmp_dir, 'model.pb')
            self.argv.input_model_is_text = False
            self.argv.output_dir = tmp_dir
            self.assertTrue(os.path.exists(self.argv.input_model),
                            "The test model in frozen binary format must exist")
            # test convert_to_pb - expect no auxiliary model created
            self.assertIsNone(convert_to_pb(self.argv))
