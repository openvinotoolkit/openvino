# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from common.tf_layer_test_class import CommonTFLayerTest


class TestNestedWhile(CommonTFLayerTest):
    def create_simple_while(self):
        import tensorflow as tf

        g = tf.Graph()
        with g.as_default():
            x = tf.compat.v1.placeholder(tf.float32, shape=(3, 2))
            v = tf.constant([1, 2, 3], dtype=tf.int32, shape=[3])
            i = tf.constant([0], dtype=tf.int32, shape=[1])
            a_combined = tf.zeros([1, 2], dtype=tf.float32)
            b_combined = tf.zeros([1, 2], dtype=tf.float32)

            def body(x_arg, v_arg, i_arg, a_combined_arg, b_combined_arg):
                x_slice = tf.slice(x_arg, [0, 0], [1, x_arg.shape[1]])
                i_arg = tf.add(i_arg, 1)
                a_combined_arg = tf.add(a_combined_arg, x_slice)
                return x_arg, v_arg, i_arg, a_combined_arg, b_combined_arg

            while_condition = lambda x, v, i, a_combined, b_combined: i < v.shape[0]

            tf.while_loop(while_condition, body, [x, v, i, a_combined, b_combined],
                          name="while_node")

        return g, None

    def create_nested_while(self):
        import tensorflow as tf

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
                          name="while_node")

        return g, None

    @pytest.mark.nightly
    def test_simple_while(self, ie_device, precision, ir_version, temp_dir, use_legacy_frontend):
        self._test(*self.create_simple_while(), ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)

    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_nested_while(self, ie_device, precision, ir_version, temp_dir, use_legacy_frontend):
        if ie_device == 'GPU':
            pytest.skip("loop:while_0 : outer input 'less:Less0' does not have primitive map issue on GPU")
        self._test(*self.create_nested_while(), ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
