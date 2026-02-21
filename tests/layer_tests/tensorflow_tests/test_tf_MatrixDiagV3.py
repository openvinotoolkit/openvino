import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest

# Test Cases for k=0 (Main Diagonal)
class TestMatrixDiagV3(CommonTFLayerTest):
    def _prepare_input(self, inputs_dict):
        for name in inputs_dict.keys():
            inputs_dict[name] = inputs_dict[name].astype(self.input_type)
        return inputs_dict

    def create_matrix_diag_v3_net(self, diagonal_shape, k, num_rows, num_cols, padding_value, align, input_type):
        self.input_type = input_type
        tf.compat.v1.reset_default_graph()


        with tf.compat.v1.Session() as sess:
            diagonal = tf.compat.v1.placeholder(input_type, diagonal_shape, 'diagonal')


            def make_const(val, dtype, name):
                return tf.constant(val, dtype=dtype, name=name)

            k_in = make_const(k, tf.int32, 'k')
            num_rows_in = make_const(num_rows, tf.int32, 'num_rows')
            num_cols_in = make_const(num_cols, tf.int32, 'num_cols')
            padding_in = make_const(padding_value, input_type, 'padding_value')

            tf.raw_ops.MatrixDiagV3(
                diagonal=diagonal,
                k=k_in,
                num_rows=num_rows_in,
                num_cols=num_cols_in,
                padding_value=padding_in,
                align=align,
                name='MatrixDiagV3'
            )

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    # Test Cases for k=0 (Main Diagonal)
    test_data_k0 = [

        (dict(diagonal_shape=[3], k=0, num_rows=-1, num_cols=-1, padding_value=0, align="RIGHT_LEFT"), np.float32),


        (dict(diagonal_shape=[2, 4], k=0, num_rows=-1, num_cols=-1, padding_value=0, align="LEFT_RIGHT"), np.int32),


        (dict(diagonal_shape=[3], k=0, num_rows=3, num_cols=3, padding_value=1.5, align="RIGHT_LEFT"), np.float32),


        (dict(diagonal_shape=[3], k=0, num_rows=3, num_cols=4, padding_value=9, align="RIGHT_LEFT"), np.int32),


        (dict(diagonal_shape=[3], k=0, num_rows=4, num_cols=3, padding_value=0, align="LEFT_RIGHT"), np.float32),
    ]

    @pytest.mark.parametrize("params, input_type", test_data_k0)
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_matrix_diag_v3_k0(self, params, input_type, ie_device, precision, ir_version, temp_dir,
                               use_legacy_frontend):
        self._test(
            *self.create_matrix_diag_v3_net(**params, input_type=input_type),
            ie_device, precision, ir_version, temp_dir=temp_dir,
            use_legacy_frontend=use_legacy_frontend)


