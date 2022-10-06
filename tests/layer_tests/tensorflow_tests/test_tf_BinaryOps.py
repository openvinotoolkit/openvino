# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from common.tf_layer_test_class import CommonTFLayerTest
from common.utils.tf_utils import permute_nchw_to_nhwc


def generate_input(op_type, size):
    narrow_borders = ["Pow"]

    logical_type = ['LogicalAnd', 'LogicalOr', 'LogicalXor']

    # usual function domain
    lower = -256
    upper = 256

    # specific domains
    if op_type in narrow_borders:
        lower = 0
        upper = 16

    if op_type in logical_type:
        return np.random.randint(0, 1, size).astype(np.bool)
    elif op_type in narrow_borders:
        return np.random.uniform(lower, upper, size).astype(np.float32)
    else:
        return np.random.uniform(lower, upper, size).astype(np.float32)


class TestBinaryOps(CommonTFLayerTest):
    def _prepare_input(self, inputs_dict):
        for input in inputs_dict.keys():
            inputs_dict[input] = generate_input(self.current_op_type, inputs_dict[input])
        return inputs_dict

    def create_add_placeholder_const_net(self, x_shape, y_shape, ir_version, op_type,
                                         use_new_frontend):
        """
            Tensorflow net                       IR net

            Placeholder->BinaryOp       =>       Placeholder->BinaryOp
                         /                                     /
            Const-------/                         Const-------/

        """

        self.current_op_type = op_type

        import tensorflow as tf

        op_type_to_tf = {
            'Add': tf.math.add,
            'Sub': tf.math.subtract,
            'Mul': tf.math.multiply,
            'Div': tf.math.divide,
            'RealDiv': tf.realdiv,
            'SquaredDifference': tf.math.squared_difference,
            'Pow': tf.math.pow,
            'Maximum': tf.math.maximum,
            'Minimum': tf.math.minimum,
            'Equal': tf.math.equal,
            'NotEqual': tf.math.not_equal,
            'Mod': tf.math.mod,
            'Greater': tf.math.greater,
            'GreaterEqual': tf.math.greater_equal,
            'Less': tf.math.less,
            'LessEqual': tf.math.less_equal,
            'LogicalAnd': tf.math.logical_and,
            'LogicalOr': tf.math.logical_or,
            'LogicalXor': tf.math.logical_xor,
            'FloorMod': tf.math.floormod,
        }

        type = np.float32
        if op_type in ["LogicalAnd", "LogicalOr", "LogicalXor"]:
            type = np.bool
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            tf_x_shape = x_shape.copy()
            tf_y_shape = y_shape.copy()

            tf_x_shape = permute_nchw_to_nhwc(tf_x_shape, use_new_frontend)
            tf_y_shape = permute_nchw_to_nhwc(tf_y_shape, use_new_frontend)

            x = tf.compat.v1.placeholder(type, tf_x_shape, 'Input')
            constant_value = generate_input(op_type, tf_y_shape)
            if (constant_value == 0).all():
                # Avoid elimination of the layer from IR
                constant_value = constant_value + 1
            y = tf.constant(constant_value, dtype=type)

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

    test_data_precommits = [dict(x_shape=[2, 3, 4], y_shape=[2, 3, 4]),
                            pytest.param(dict(x_shape=[2, 3, 4, 5], y_shape=[2, 3, 4, 5]),
                                         marks=pytest.mark.precommit_tf_fe)]

    @pytest.mark.parametrize("params", test_data_precommits)
    @pytest.mark.parametrize("op_type",
                             ['Add', 'Sub', 'Mul', 'Div', 'RealDiv', 'SquaredDifference', 'Pow',
                              'Maximum', 'Minimum',
                              'Equal', 'NotEqual', 'Mod', 'Greater', 'GreaterEqual', 'Less',
                              'LessEqual',
                              'LogicalAnd', 'LogicalOr', 'LogicalXor', 'FloorMod'])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_binary_op(self, params, ie_device, precision, ir_version, temp_dir, op_type,
                       use_new_frontend, use_old_api):
        if ie_device == 'GPU' and precision == "FP16":
            pytest.skip("BinaryOps tests temporary skipped on GPU with FP16 precision."
                        "Several tests don't pass accuracy checks.")
        self._test(
            *self.create_add_placeholder_const_net(**params, ir_version=ir_version, op_type=op_type,
                                                   use_new_frontend=use_new_frontend), ie_device,
            precision,
            ir_version, temp_dir=temp_dir, use_new_frontend=use_new_frontend, use_old_api=use_old_api)
