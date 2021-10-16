# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from common.tf_layer_test_class import CommonTFLayerTest
from common.utils.tf_utils import permute_nchw_to_nhwc


class TestAdd(CommonTFLayerTest):
    def create_add_placeholder_const_net(self, x_shape, y_shape, ir_version, op_type, use_new_frontend):
        """
            Tensorflow net                  IR net

            Placeholder->Add       =>       Placeholder->Eltwise or Power or ScaleShift
                         /                               /
            Const-------/                   Const-------/

        """

        #
        #   Create Tensorflow model
        #
        import tensorflow as tf

        op_type_to_tf = {
            'Add': tf.add,
            'Sub': tf.subtract,
            'Mul': tf.multiply,
            'Div': tf.divide,
            'RealDiv': tf.realdiv,
            'SquaredDifference': tf.squared_difference,
            'Pow': tf.pow,
            'Maximum': tf.maximum,
            'Minimum': tf.minimum,
            'Equal': tf.equal,
            'NotEqual': tf.not_equal,
            'Mod': tf.mod,
            'Greater': tf.greater,
            'GreaterEqual': tf.greater_equal,
            'Less': tf.less,
            'LessEqual': tf.less_equal,
            'LogicalAnd': tf.logical_and,
            'LogicalOr': tf.logical_or,
            'LogicalXor': tf.logical_xor,
            'FloorMod': tf.floormod,
        }

        tf.compat.v1.reset_default_graph()

        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            tf_x_shape = x_shape.copy()
            tf_y_shape = y_shape.copy()

            tf_x_shape = permute_nchw_to_nhwc(tf_x_shape, use_new_frontend)
            tf_y_shape = permute_nchw_to_nhwc(tf_y_shape, use_new_frontend)

            x = tf.compat.v1.placeholder(tf.float32, tf_x_shape, 'Input')
            constant_value = np.random.randint(-256, 256, tf_y_shape).astype(np.float32)
            if (constant_value == 0).all():
                # Avoid elimination of the layer from IR
                constant_value = constant_value + 1
            y = tf.constant(constant_value)

            op = op_type_to_tf[op_type](x, y, name="Operation")

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        #
        #   Create reference IR net
        #   Please, specify 'type': 'Input' for input node
        #   Moreover, do not forget to validate ALL layer attributes!!!
        #

        ref_net = None

        return tf_net, ref_net

    # TODO: implement tests for 2 Consts + Add

    test_data_1D = [
        # Power
        dict(x_shape=[1], y_shape=[1]),
        # Eltwise
        pytest.param(dict(x_shape=[3], y_shape=[3]))
    ]

    @pytest.mark.parametrize("params", test_data_1D)
    @pytest.mark.parametrize("op_type", [
        'Add',
      'Sub',
    'Mul',
    'Div',
    'RealDiv',
    'SquaredDifference',
    'Pow',
    'Maximum',
    'Minimum',
    'Equal',
    'NotEqual',
    'Mod',
    'Greater',
    'GreaterEqual',
    'Less',
    'LessEqual',
    'LogicalAnd',
    'LogicalOr',
    'LogicalXor',
    'FloorMod',
    ])
    @pytest.mark.nightly
    def test_add_placeholder_const_1D(self, params, ie_device, precision, ir_version, temp_dir, op_type,
                                      use_new_frontend):
        self._test(*self.create_add_placeholder_const_net(**params, ir_version=ir_version, op_type=op_type,
                                                          use_new_frontend=use_new_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir, use_new_frontend=use_new_frontend)


