# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from common.tf_layer_test_class import CommonTFLayerTest


class TestLoop(CommonTFLayerTest):
    skip_framework = True

    def create_loop(self):
        """
            TF net

            Input->Loop->Output   =>   Only accuracy check

        """

        #   Create TF model
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.compat.v1 import graph_util
        from tensorflow.python.keras import backend as K
        from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

        g = tf.Graph()
        with g.as_default():
            x = tf.compat.v1.placeholder(tf.float32, shape=(3, 2))
            v = tf.constant([1, 2, 3], dtype=tf.int32, shape=[3])
            i = tf.constant([0], dtype=tf.int32, shape=[1])
            a_combined = tf.zeros([1, 2], dtype=tf.float32)
            b_combined = tf.zeros([1, 2], dtype=tf.float32)

            def body(x_arg, v_arg, i_arg, a_combined_arg, b_combined_arg):
                x_slice = tf.slice(x_arg, [0, 0], [1, x_arg.shape[1]])
                v_slice = tf.slice(v_arg, [0], [1])
                j = tf.constant([0], dtype=tf.int32, shape=[1])
                i_arg = tf.add(i_arg, 1)
                a_combined_arg = tf.add(a_combined_arg, x_slice)
                return x_arg, v_arg, i_arg, a_combined_arg, b_combined_arg

            while_condition = lambda x, v, i, a_combined, b_combined: i < v.shape[0]

            tf.while_loop(while_condition, body, [x, v, i, a_combined, b_combined], name="whilenode")

        return g, None

    def create_loop_in_loop(self):
        """
            TF net

            Input->Loop(Loop)->Output   =>   Only accuracy check

        """
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.compat.v1 import graph_util
        from tensorflow.python.keras import backend as K
        from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

        #   Create TF model

        g = tf.Graph()
        with g.as_default():
            x = tf.compat.v1.placeholder(tf.float32, shape=(3, 2))
            v = tf.constant([1, 2, 3], dtype=tf.int32, shape=[3])
            i = tf.constant([0], dtype=tf.int32, shape=[1])
            a_combined = tf.zeros([1, 2], dtype=tf.float32)
            b_combined = tf.zeros([1, 2], dtype=tf.float32)

            def body(x_arg, v_arg, i_arg, a_combined_arg, b_combined_arg):
                x_slice = tf.slice(x_arg, [0, 0], [1, x_arg.shape[1]])
                v_slice = tf.slice(v_arg, [0], [1])
                j = tf.constant([0], dtype=tf.int32, shape=[1])

                def body_supp(x_slice_arg, v_slice_arg, j_arg, b_combined_arg_arg):
                    j_arg = tf.add(j_arg, 1)
                    b_combined_arg_arg = tf.add(b_combined_arg_arg, x_slice_arg)
                    return x_slice_arg, v_slice_arg, j_arg, b_combined_arg_arg

                while_condition_supp = lambda x_slice, v_slice, j, b_combined: tf.less(j, v_slice)

                x_slice, v_slice, j, b_combined_arg = tf.while_loop(while_condition_supp, body_supp,
                                                                    [x_slice, v_slice, j, b_combined_arg])

                i_arg = tf.add(i_arg, 1)

                a_combined_arg = tf.add(a_combined_arg, x_slice)
                return x_arg, v_arg, i_arg, a_combined_arg, b_combined_arg

            while_condition = lambda x, v, i, a_combined, b_combined: i < v.shape[0]

            tf.while_loop(while_condition, body, [x, v, i, a_combined, b_combined],
                                                            name="whilenode")

        return g, None

    @pytest.mark.precommit
    @pytest.mark.timeout(250)
    def test_loop_simple_precommit(self, ie_device, precision, ir_version, temp_dir):
        if ie_device == 'GPU':
            pytest.skip('Loop not supported on GPU')
        self._test(*self.create_loop(), ie_device, precision, ir_version, temp_dir=temp_dir,
                   infer_timeout=150)

    @pytest.mark.precommit
    @pytest.mark.timeout(250)
    def test_loop_in_loop_simple_precommit(self, ie_device, precision, ir_version, temp_dir):
        if ie_device == 'GPU':
            pytest.skip('Loop not supported on GPU')
        self._test(*self.create_loop_in_loop(), ie_device, precision, ir_version, temp_dir=temp_dir)
