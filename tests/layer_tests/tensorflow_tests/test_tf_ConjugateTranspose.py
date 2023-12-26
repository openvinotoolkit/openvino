# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
from common.tf_layer_test_class import CommonTFLayerTest

# Testing operation ConjugateTranspose
# Documentation: https://www.tensorflow.org/api_docs/python/tf/raw_ops/ConjugateTranspose


class TestConjugateTranspose(CommonTFLayerTest):
    
    def _prepare_input(self, inputs_info):
        
        assert 'x' in inputs_info
        input_shape = inputs_info['x']
        inputs_data = {}
        inputs_data['x'] = np.random.default_rng().random(input_shape).astype(np.complex64)
        return inputs_data
    
    def create_conjugate_transpose_net(self, shape, perm):
        """
            TensorFlow net                                 IR net

            Placeholder->ConjugateTranspose    =>       Placeholder->Transpose->Conjugate->Transpose
        """

        import tensorflow as tf

        tf.compat.v1.reset_default_graph()

        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            tf_input = tf.compat.v1.placeholder(tf.complex64, shape, "Input")

            tf.raw_ops.ConjugateTranspose(x=tf_input, perm=perm, name = "Operation")

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        ref_net = None

        return tf_net, ref_net
    
    
    test_data = [
        (dict(shape=[1, 2], perm=[1, 0])),
        (dict(shape=[1, 2, 3], perm=[2, 1, 0])),
        (dict(shape=[1, 2, 3, 4], perm=[0, 3, 2, 1])),
        (dict(shape=[1, 2, 3, 4, 5, 6], perm=[0, 2, 1, 3, 4, 5])),
    ]
    
    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.precommit_tf_fe
    @pytest.mark.nightly
    def test_conjugate_transpose(self, params, ie_device, precision, ir_version, temp_dir,
                                 use_new_frontend, use_old_api):
        self._test(*self.create_conjugate_transpose_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)