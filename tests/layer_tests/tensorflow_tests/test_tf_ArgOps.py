import pytest

from common.tf_layer_test_class import CommonTFLayerTest


class TestTFArgOps(CommonTFLayerTest):
    def create_net(self, input_shape, axis, op, ir_version):
        import tensorflow as tf
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            tf_input_shape = input_shape.copy()
            tf_axis = axis
            input_node = tf.compat.v1.placeholder(tf.float32, tf_input_shape, 'Input')
            if op == "ArgMin":
                tf.math.argmin(input_node, axis=tf_axis, name='argmin')
            else:
                tf.math.argmax(input_node, axis=tf_axis, name='argmax')
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        ref_net = None
        return tf_net, ref_net

    precommit_test_data = [
        dict(input_shape=[2, 3, 4, 5], axis=2, op='ArgMin'),
        dict(input_shape=[2, 3, 4, 5], axis=3, op='ArgMin'),
        dict(input_shape=[2, 3, 4, 5], axis=-1, op='ArgMin'),
        dict(input_shape=[2, 3, 4, 5], axis=2, op='ArgMax'),
        dict(input_shape=[2, 3, 4, 5], axis=3, op='ArgMax'),
        dict(input_shape=[2, 3, 4, 5], axis=-1, op='ArgMax')
    ]

    @pytest.mark.parametrize("params", precommit_test_data)
    @pytest.mark.precommit
    def test_tf_argmin_precommit(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    test_data = [
        dict(input_shape=[2], axis=0, op='ArgMin'),
        dict(input_shape=[2], axis=0, op='ArgMax'),
        dict(input_shape=[2, 3], axis=0, op='ArgMin'),
        dict(input_shape=[2, 3], axis=1, op='ArgMin'),
        dict(input_shape=[2, 3], axis=0, op='ArgMax'),
        dict(input_shape=[2, 3], axis=1, op='ArgMax'),
        dict(input_shape=[2, 3, 4], axis=0, op='ArgMin'),
        dict(input_shape=[2, 3, 4], axis=1, op='ArgMin'),
        dict(input_shape=[2, 3, 4], axis=2, op='ArgMin'),
        dict(input_shape=[2, 3, 4], axis=0, op='ArgMax'),
        dict(input_shape=[2, 3, 4], axis=1, op='ArgMax'),
        dict(input_shape=[2, 3, 4], axis=2, op='ArgMax'),
        dict(input_shape=[2, 3, 4, 5], axis=0, op="ArgMin"),
        dict(input_shape=[2, 3, 4, 5], axis=1, op="ArgMin"),
        dict(input_shape=[2, 3, 4, 5], axis=2, op="ArgMin"),
        dict(input_shape=[2, 3, 4, 5], axis=3, op="ArgMin"),
        dict(input_shape=[2, 3, 4, 5], axis=0, op="ArgMax"),
        dict(input_shape=[2, 3, 4, 5], axis=1, op="ArgMax"),
        dict(input_shape=[2, 3, 4, 5], axis=2, op="ArgMax"),
        dict(input_shape=[2, 3, 4, 5], axis=3, op="ArgMax"),
        dict(input_shape=[2, 3, 4, 5, 6], axis=0, op='ArgMin'),
        dict(input_shape=[2, 3, 4, 5, 6], axis=1, op='ArgMin'),
        dict(input_shape=[2, 3, 4, 5, 6], axis=2, op='ArgMin'),
        dict(input_shape=[2, 3, 4, 5, 6], axis=3, op='ArgMin'),
        dict(input_shape=[2, 3, 4, 5, 6], axis=4, op='ArgMin'),
        dict(input_shape=[2, 3, 4, 5, 6], axis=0, op='ArgMax'),
        dict(input_shape=[2, 3, 4, 5, 6], axis=1, op='ArgMax'),
        dict(input_shape=[2, 3, 4, 5, 6], axis=2, op='ArgMax'),
        dict(input_shape=[2, 3, 4, 5, 6], axis=3, op='ArgMax'),
        dict(input_shape=[2, 3, 4, 5, 6], axis=4, op='ArgMax'),
        dict(input_shape=[2, 3, 4, 5, 6], axis=-1, op='ArgMin'),
        dict(input_shape=[2, 3, 4, 5, 6], axis=-2, op='ArgMin'),
        dict(input_shape=[2, 3, 4, 5, 6], axis=-3, op='ArgMin'),
        dict(input_shape=[2, 3, 4, 5, 6], axis=-4, op='ArgMin'),
        dict(input_shape=[2, 3, 4, 5, 6], axis=-5, op='ArgMin'),
        dict(input_shape=[2, 3, 4, 5, 6], axis=-1, op='ArgMax'),
        dict(input_shape=[2, 3, 4, 5, 6], axis=-2, op='ArgMax'),
        dict(input_shape=[2, 3, 4, 5, 6], axis=-3, op='ArgMax'),
        dict(input_shape=[2, 3, 4, 5, 6], axis=-4, op='ArgMax'),
        dict(input_shape=[2, 3, 4, 5, 6], axis=-5, op='ArgMax'),
    ]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_tf_argmin(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)
