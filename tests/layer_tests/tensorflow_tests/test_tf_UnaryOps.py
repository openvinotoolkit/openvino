# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import sys
from common.layer_test_class import check_ir_version
from common.tf_layer_test_class import CommonTFLayerTest
from common.utils.tf_utils import permute_nchw_to_nhwc

from unit_tests.utils.graph import build_graph


class TestUnaryOps(CommonTFLayerTest):
    current_op_type = None

    def _prepare_input(self, inputs_dict):
        non_negative = ['Sqrt', 'Log']
        narrow_borders = ["Sinh", "Cosh", "Tanh", "Exp"]
        within_one = ['Asin', 'Acos', 'Atanh']
        from_one = ['Acosh']

        logical_type = ['LogicalNot']

        # usual function domain
        lower = -256
        upper = 256

        # specific domains
        if self.current_op_type in non_negative:
            lower = 0
        elif self.current_op_type in narrow_borders:
            lower = -16
            upper = 16
        elif self.current_op_type in from_one:
            lower = 1
        elif self.current_op_type in within_one:
            lower = -1
            upper = 1

        for input in inputs_dict.keys():
            if self.current_op_type in logical_type:
                inputs_dict[input] = np.random.randint(0, 1, inputs_dict[input]).astype(bool)
            else:
                inputs_dict[input] = np.random.uniform(lower, upper, inputs_dict[input]).astype(
                    np.float32)

        return inputs_dict

    def create_net_with_mish(self, shape, ir_version, use_new_frontend):
        """
        TODO: Move functionality to `create_net_with_unary_op()` once tensorflow_addons
        supports Python 3.11
            Tensorflow net                 IR net

            Input->mish          =>       Input->mish
        """
        import tensorflow as tf
        import tensorflow_addons as tfa

        tf.compat.v1.reset_default_graph()

        with tf.compat.v1.Session() as sess:
            tf_x_shape = shape.copy()
            tf_x_shape = permute_nchw_to_nhwc(tf_x_shape, use_new_frontend)

            input = tf.compat.v1.placeholder(tf.float32, tf_x_shape, 'Input')
            tfa.activations.mish(input)

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        #
        #   Create reference IR net
        #   Please, specify 'type': 'Input' for input node
        #   Moreover, do not forget to validate ALL layer attributes!!!
        #

        ref_net = None

        if check_ir_version(10, None, ir_version) and not use_new_frontend:
            nodes_attributes = {
                'input': {'kind': 'op', 'type': 'Parameter'},
                'input_data': {'shape': shape, 'kind': 'data'},
                'testing_op': {'kind': 'op', 'type': 'Mish'},
                'testing_data': {'shape': shape, 'kind': 'data'},
                'result': {'kind': 'op', 'type': 'Result'}
            }

            ref_net = build_graph(nodes_attributes,
                                  [('input', 'input_data'),
                                   ('input_data', 'testing_op'),
                                   ('testing_op', 'testing_data'),
                                   ('testing_data', 'result')
                                   ])

        return tf_net, ref_net

    def create_net_with_unary_op(self, shape, ir_version, op_type, use_new_frontend):
        """
        TODO: Move functionality of `create_net_with_mish()` here once tensorflow_addons
        supports Python 3.11
            Tensorflow net                 IR net

            Input->UnaryOp       =>       Input->UnaryOp

        """
        import tensorflow as tf

        self.current_op_type = op_type
        op_type_to_tf = {
            'Abs': tf.math.abs,
            'Acos': tf.math.acos,
            'Acosh': tf.math.acosh,
            'Asin': tf.math.asin,
            'Asinh': tf.math.asinh,
            'Atan': tf.math.atan,
            'Atanh': tf.math.atanh,
            'Ceiling': tf.math.ceil,
            'Cos': tf.math.cos,
            'Cosh': tf.math.cosh,
            'Elu': tf.nn.elu,
            'Erf': tf.math.erf,
            'Exp': tf.math.exp,
            'Floor': tf.math.floor,
            'Log': tf.math.log,
            'LogicalNot': tf.math.logical_not,
            #'Mish': tfa.activations.mish,  # temporarily moved to `create_net_with_mish()`
            'Negative': tf.math.negative,
            'Sigmoid': tf.nn.sigmoid,
            'Sign': tf.math.sign,
            'Sin': tf.math.sin,
            'Sinh': tf.math.sinh,
            'SoftPlus': tf.nn.softplus,
            'Square': tf.math.square,
            'Tan': tf.math.tan,
            'Tanh': tf.math.tanh,
            'ReLU': tf.nn.relu,
        }

        tf.compat.v1.reset_default_graph()

        type = tf.float32
        if op_type == "LogicalNot":
            type = tf.bool
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            tf_x_shape = shape.copy()
            tf_x_shape = permute_nchw_to_nhwc(tf_x_shape, use_new_frontend)

            input = tf.compat.v1.placeholder(type, tf_x_shape, 'Input')
            if self.current_op_type == 'Mish':
                # Mish has no attribute name
                op_type_to_tf[self.current_op_type](input)
            else:
                op_type_to_tf[self.current_op_type](input, name='Operation')

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        #
        #   Create reference IR net
        #   Please, specify 'type': 'Input' for input node
        #   Moreover, do not forget to validate ALL layer attributes!!!
        #

        ref_net = None

        if check_ir_version(10, None, ir_version) and not use_new_frontend:
            nodes_attributes = {
                'input': {'kind': 'op', 'type': 'Parameter'},
                'input_data': {'shape': shape, 'kind': 'data'},
                'testing_op': {'kind': 'op', 'type': self.current_op_type},
                'testing_data': {'shape': shape, 'kind': 'data'},
                'result': {'kind': 'op', 'type': 'Result'}
            }

            ref_net = build_graph(nodes_attributes,
                                  [('input', 'input_data'),
                                   ('input_data', 'testing_op'),
                                   ('testing_op', 'testing_data'),
                                   ('testing_data', 'result')
                                   ])

        return tf_net, ref_net

    test_data_precommit = [dict(shape=[4, 6, 8, 10, 12])]

    @pytest.mark.parametrize("params", test_data_precommit)
    @pytest.mark.parametrize("op_type", ['Elu',
                                         'Sigmoid',
                                         'Sin',
                                         'Sinh',
                                         'Cos',
                                         'Cosh',
                                         'Abs',
                                         'Negative',
                                         'Exp',
                                         'Tan',
                                         'Tanh',
                                         'Floor',
                                         'ReLU',
                                         'Ceiling',
                                         'Asin',
                                         'Acos',
                                         'Atan',
                                         'Log',
                                         'Sign',
                                         'SoftPlus',
                                         'Atanh',
                                         'Acosh',
                                         'Asinh',
                                         'LogicalNot',
                                         'Square',
                                         'Erf',
                                         ])
    @pytest.mark.precommit
    def test_unary_op_precommit(self, params, ie_device, precision, ir_version, temp_dir, op_type,
                                use_new_frontend, use_old_api):
        if ie_device == 'GPU':
            pytest.skip("5D tensors is not supported on GPU")
        self._test(*self.create_net_with_unary_op(**params, ir_version=ir_version, op_type=op_type,
                                                  use_new_frontend=use_new_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)
        
    @pytest.mark.xfail(sys.version_info > (3, 10), reason="tensorflow_addons package is not available for Python 3.11 and higher")
    @pytest.mark.parametrize("params", test_data_precommit)
    @pytest.mark.precommit
    def test_unary_op_mish_precommit(self, params, ie_device, precision, ir_version, temp_dir,
                                use_new_frontend, use_old_api):
        """
        TODO: Move to `test_unary_op_precommit()` once tensorflow_addons package is available for Python 3.11
        """
        if ie_device == 'GPU':
            pytest.skip("5D tensors is not supported on GPU")
        self._test(*self.create_net_with_mish(**params, ir_version=ir_version,
                                              use_new_frontend=use_new_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)

    test_data = [pytest.param(dict(shape=[10, 12]), marks=pytest.mark.precommit_tf_fe),
                 dict(shape=[8, 10, 12]),
                 dict(shape=[6, 8, 10, 12]),
                 dict(shape=[4, 6, 8, 10, 12])]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.parametrize("op_type", ['Elu',
                                         'Sigmoid',
                                         'Sin',
                                         'Sinh',
                                         'Cos',
                                         'Cosh',
                                         'Abs',
                                         'Negative',
                                         'Exp',
                                         'Tan',
                                         'Tanh',
                                         'Floor',
                                         'ReLU',
                                         'Ceiling',
                                         'Asin',
                                         'Acos',
                                         'Atan',
                                         'Log',
                                         'LogicalNot',
                                         'Sign',
                                         'SoftPlus',
                                         'Atanh',
                                         'Acosh',
                                         'Asinh',
                                         'Square',
                                         'Erf',
                                         ])
    @pytest.mark.nightly
    def test_unary_op(self, params, ie_device, precision, ir_version, temp_dir, op_type,
                      use_new_frontend, use_old_api):
        if ie_device == 'GPU':
            pytest.skip("5D tensors is not supported on GPU")
        self._test(*self.create_net_with_unary_op(**params, ir_version=ir_version, op_type=op_type,
                                                  use_new_frontend=use_new_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)
        
    @pytest.mark.xfail(sys.version_info > (3, 10), reason="tensorflow_addons package is not available for Python 3.11 and higher")
    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_unary_op_mish(self, params, ie_device, precision, ir_version, temp_dir, op_type,
                           use_new_frontend, use_old_api):
        """
        TODO: Move to `test_unary_op()` once tensorflow_addons package is available for Python 3.11
        """
        if ie_device == 'GPU':
            pytest.skip("5D tensors is not supported on GPU")
        self._test(*self.create_net_with_mish(**params, ir_version=ir_version,
                                              use_new_frontend=use_new_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)
