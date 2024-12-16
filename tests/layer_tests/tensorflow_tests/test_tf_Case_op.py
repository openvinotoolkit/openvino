import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


class TestCaseOp(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        """
        Prepares input data based on the given input shapes and data types.
        """
        assert 'cond' in inputs_info
        assert 'input_data' in inputs_info
        inputs_data = {
            'cond': np.array(inputs_info['cond'], dtype=np.bool_),
            'input_data': np.array(inputs_info['input_data'], dtype=np.float32)
        }
        return inputs_data

    def create_case_net(self, input_shape, cond_value):
        """
        Creates a TensorFlow model with a Case operation.

        Args:
            input_shape: Shape of the input tensor.
            cond_value: The condition value to select the branch.

        Returns:
            TensorFlow graph definition and None.
        """
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            # Inputs
            cond = tf.compat.v1.placeholder(dtype=tf.bool, shape=(), name="cond")
            input_data = tf.compat.v1.placeholder(dtype=tf.float32, shape=input_shape, name="input_data")
            
            # Define branch functions
            def branch_fn_1():
                return tf.add(input_data, tf.constant(1.0, dtype=tf.float32))

            def branch_fn_2():
                return tf.multiply(input_data, tf.constant(2.0, dtype=tf.float32))

            branches_fn = [branch_fn_1, branch_fn_2]

            # Create Case operation
            case_op = tf.raw_ops.Case(branch_index=cond, branches=branches_fn, output_type=tf.float32)
            tf.identity(case_op, name="output")

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    # Test parameters
    test_data_basic = [
        dict(input_shape=[1, 2], cond=True),
        dict(input_shape=[3, 3], cond=False),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit_tf_fe
    @pytest.mark.nightly
    def test_case_op(self, params, ie_device, precision, ir_version, temp_dir,
                     use_new_frontend, use_old_api):
        """
        Executes the test for the Case operation.
        """
        self._test(*self.create_case_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)
