# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import platform

import numpy as np
import pytest
from common.tf_layer_test_class import CommonTFLayerTest


def generate_input(op_type, size):
    narrow_borders = ["Pow"]

    logical_type = ['LogicalAnd', 'LogicalOr', 'LogicalXor']

    bitwise_type = ['BitwiseAnd', 'BitwiseOr', 'BitwiseXor']

    # usual function domain
    lower = -256
    upper = 256

    # specific domains
    if op_type in narrow_borders:
        lower = 0
        upper = 16

    if op_type in logical_type:
        return np.random.randint(0, 1, size).astype(bool)
    elif op_type in bitwise_type:
        return np.random.randint(lower, upper, size).astype(np.int32)
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
                                         use_legacy_frontend):
        if not use_legacy_frontend and op_type == "Xdivy":
            pytest.xfail(reason="95499")

        self.current_op_type = op_type

        import tensorflow as tf

        op_type_to_tf = {
            'Add': tf.math.add,
            'AddV2': tf.raw_ops.AddV2,
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
            'FloorDiv': tf.math.floordiv,
            'Xdivy': tf.raw_ops.Xdivy,
            'BitwiseAnd': tf.raw_ops.BitwiseAnd,
            'BitwiseOr': tf.raw_ops.BitwiseOr,
            'BitwiseXor': tf.raw_ops.BitwiseXor,
        }

        op_type_kw_args = ["AddV2", "Xdivy", "BitwiseAnd", "BitwiseOr", "BitwiseXor"]

        type = np.float32
        if op_type in ["LogicalAnd", "LogicalOr", "LogicalXor"]:
            type = bool
        elif op_type in ["BitwiseAnd", "BitwiseOr", "BitwiseXor"]:
            type = np.int32
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(type, x_shape, 'Input')
            constant_value = generate_input(op_type, y_shape)
            if (constant_value == 0).all():
                # Avoid elimination of the layer from IR
                constant_value = constant_value + 1
            y = tf.constant(constant_value, dtype=type)

            if not op_type in op_type_kw_args:
                op = op_type_to_tf[op_type](x, y, name="Operation")
            else:
                op = op_type_to_tf[op_type](x=x, y=y, name="Operation")

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        ref_net = None

        return tf_net, ref_net

    test_data_precommits = [dict(x_shape=[2, 3, 4], y_shape=[2, 3, 4]),
                            pytest.param(dict(x_shape=[2, 3, 4, 5], y_shape=[2, 3, 4, 5]),
                                         marks=pytest.mark.precommit_tf_fe)]

    @pytest.mark.parametrize("params", test_data_precommits)
    @pytest.mark.parametrize("op_type",
                             ['Add', 'AddV2', 'Sub', 'Mul', 'Div', 'RealDiv', 'SquaredDifference', 'Pow',
                              'Maximum', 'Minimum',
                              'Equal', 'NotEqual', 'Mod', 'Greater', 'GreaterEqual', 'Less',
                              'LessEqual',
                              'LogicalAnd', 'LogicalOr', 'LogicalXor', 'FloorMod', 'FloorDiv',
                              'Xdivy', 'BitwiseAnd', 'BitwiseOr', 'BitwiseXor', ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.xfail(condition=platform.system() == 'Darwin' and platform.machine() == 'arm64',
                       reason='Ticket - 122716')
    def test_binary_op(self, params, ie_device, precision, ir_version, temp_dir, op_type,
                       use_legacy_frontend):
        if not use_legacy_frontend and op_type in ['BitwiseAnd', 'BitwiseOr', 'BitwiseXor']:
            pytest.skip("Bitwise ops are supported only by new TF FE.")
        if precision == "FP16":
            pytest.skip("BinaryOps tests are skipped with FP16 precision."
                        "They don't pass accuracy checks because chaotic output.")
        self._test(
            *self.create_add_placeholder_const_net(**params, ir_version=ir_version, op_type=op_type,
                                                   use_legacy_frontend=use_legacy_frontend), ie_device,
            precision,
            ir_version, temp_dir=temp_dir, use_legacy_frontend=use_legacy_frontend)
