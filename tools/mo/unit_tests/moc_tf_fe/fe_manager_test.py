import unittest
from openvino.frontend import FrontEndManager  # pylint: disable=no-name-in-module,import-error


class TestConversionWithBatchAndLayout(unittest.TestCase):
    def test_load_by_model_tf_graph_iterator(self):
        def simple_tf_model():
            import tensorflow as tf

            tf.compat.v1.reset_default_graph()

            with tf.compat.v1.Session() as sess:
                inp = tf.compat.v1.placeholder(tf.float32, [1, 2, 3], "Input")
                _ = tf.nn.sigmoid(inp, name="Sigmoid")

                tf.compat.v1.global_variables_initializer()
                tf_net = sess.graph
            return tf_net
        from openvino.frontend.tensorflow.graph_iterator import GraphIteratorTFGraph
        model = GraphIteratorTFGraph(simple_tf_model())
        fem = FrontEndManager()
        fe = fem.load_by_model(model)
        assert fe is not None
        assert fe.get_name() == "tf"
