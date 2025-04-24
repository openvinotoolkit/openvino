# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from common.tf_layer_test_class import CommonTFLayerTest


class TestMatMul(CommonTFLayerTest):

    def create_net_with_matmul_op(self, x_shape, y_shape, x_bool, y_bool, op_type, ir_version, use_legacy_frontend):
        import tensorflow as tf
        op_type_to_tf = {
            'BatchMatMul': tf.raw_ops.BatchMatMul,
            'BatchMatMulV2': tf.raw_ops.BatchMatMulV2,
            'BatchMatMulV3': tf.raw_ops.BatchMatMulV3,
            'MatMul': tf.raw_ops.MatMul,
        }

        tf.compat.v1.reset_default_graph()

        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            tf_x = tf.compat.v1.placeholder(tf.float32, x_shape, 'InputX')
            tf_y = tf.compat.v1.placeholder(tf.float32, y_shape, 'InputY')
            if op_type == 'MatMul':
                if len(x_shape) != 2 or len(y_shape) != 2:
                    pytest.skip("MatMul doesn't support rank != 2")
                op_type_to_tf[op_type](a=tf_x, b=tf_y, transpose_a=x_bool, transpose_b=y_bool, name='Operation')
            elif op_type == 'BatchMatMul':
                if len(x_shape) != len(y_shape):
                    pytest.skip("BatchMatMul doesn't support broadcast")
                op_type_to_tf[op_type](x=tf_x, y=tf_y, adj_x=x_bool, adj_y=y_bool, name='Operation')
            elif op_type == 'BatchMatMulV2':
                op_type_to_tf[op_type](x=tf_x, y=tf_y, adj_x=x_bool, adj_y=y_bool, name='Operation')
            elif op_type == 'BatchMatMulV3':
                op_type_to_tf[op_type](x=tf_x, y=tf_y, Tout=tf.float32, adj_x=x_bool, adj_y=y_bool, name='Operation')
            else:
                raise RuntimeError("Unknown operation")

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        ref_net = None

        return tf_net, ref_net

    test_data_precommit = [
        dict(x_shape=[2, 4, 4], y_shape=[2, 4, 4]),     #Tests 2D shapes
        dict(x_shape=[2, 3, 4, 4], y_shape=[4, 4]),     #Tests broadcast
        ]

    @pytest.mark.parametrize("params", test_data_precommit)
    @pytest.mark.parametrize("op_type", ['BatchMatMul',
                                         'BatchMatMulV2',
                                         'BatchMatMulV3',
                                         'MatMul',
                                         ])
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_matmul_op_precommit(self, params, ie_device, precision, ir_version, temp_dir, op_type,
                                 use_legacy_frontend):
        self._test(*self.create_net_with_matmul_op(**params, ir_version=ir_version, op_type=op_type,
                                                  use_legacy_frontend=use_legacy_frontend, x_bool=False, y_bool=False),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)

    test_data = test_data_precommit + [
        dict(x_shape=[2, 3, 4, 4], y_shape=[2, 3, 4, 4]),   #Tests 4D shapes
        ]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.parametrize("op_type", ['BatchMatMul',
                                         'BatchMatMulV2',
                                         'BatchMatMulV3',
                                         'MatMul',
                                         ])
    @pytest.mark.parametrize("x_bool", [
        False,
        True
        ])
    @pytest.mark.parametrize("y_bool", [
        False,
        True
        ])
    @pytest.mark.nightly
    def test_matmul_op_nightly(self, params, ie_device, precision, ir_version, temp_dir, op_type,
                                x_bool, y_bool, use_legacy_frontend):
        self._test(*self.create_net_with_matmul_op(**params, ir_version=ir_version, op_type=op_type,
                                                  use_legacy_frontend=use_legacy_frontend, x_bool=x_bool, y_bool=y_bool),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
