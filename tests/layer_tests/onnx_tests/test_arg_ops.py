import pytest

from common.onnx_layer_test_class import OnnxRuntimeLayerTest


class TestOnnxArgOps(OnnxRuntimeLayerTest):
    def create_net(self, shape, axis, keepdims, op, ir_version):
        import onnx
        from onnx import helper
        from onnx import TensorProto

        output_shape = shape.copy()
        output_shape[axis] = 1
        output_shape_squeeze = output_shape.copy()
        if keepdims == 0:
            output_shape_squeeze.remove(1)
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, shape)
        output = helper.make_tensor_value_info('output', TensorProto.INT64, output_shape_squeeze)
        args = dict()
        args['axis'] = axis
        args['keepdims'] = keepdims
        if op == 'ArgMin':
            node_def = onnx.helper.make_node(
                'ArgMin',
                inputs=['input'],
                outputs=['argmin' if keepdims == 1 else 'output'],
                **args
            )
        else:
            node_def = onnx.helper.make_node(
                'ArgMax',
                inputs=['input'],
                outputs=['argmin' if keepdims == 1 else 'output'],
                **args
            )

        edges = [node_def]

        if keepdims == 1:
            node_flatten_def = onnx.helper.make_node(
                'Flatten',
                inputs=['argmin'],
                outputs=['output']
            )
            edges.append(node_flatten_def)

        # Create the graph (GraphProto)
        graph_def = helper.make_graph(
            edges,
            'test_model',
            [input],
            [output],
        )

        # Create the model (ModelProto)
        onnx_net = helper.make_model(graph_def, producer_name='test_model')
        onnx.checker.check_model(onnx_net)

        ref_net = None

        return onnx_net, ref_net

    precommit_test_data = [
        dict(shape=[2, 3, 4, 5], axis=2),
        dict(shape=[2, 3, 4, 5], axis=3),
        dict(shape=[2, 3, 4, 5], axis=-1),
    ]

    @pytest.mark.parametrize("params", precommit_test_data)
    @pytest.mark.parametrize("keepdims", [0, 1])
    @pytest.mark.parametrize('op', ['ArgMin', 'ArgMax'])
    @pytest.mark.precommit
    def test_argmin_precommit(self, params, keepdims, op, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, op=op, ir_version=ir_version, keepdims=keepdims),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    test_data = [
        dict(shape=[2], axis=0),
        dict(shape=[2, 3], axis=0),
        dict(shape=[2, 3], axis=1),
        dict(shape=[2, 3, 4], axis=0),
        dict(shape=[2, 3, 4], axis=1),
        dict(shape=[2, 3, 4], axis=2),
        dict(shape=[2, 3, 4, 5], axis=0),
        dict(shape=[2, 3, 4, 5], axis=1),
        dict(shape=[2, 3, 4, 5], axis=2),
        dict(shape=[2, 3, 4, 5], axis=3),
        dict(shape=[2, 3, 4, 5, 6], axis=0),
        dict(shape=[2, 3, 4, 5, 6], axis=1),
        dict(shape=[2, 3, 4, 5, 6], axis=2),
        dict(shape=[2, 3, 4, 5, 6], axis=3),
        dict(shape=[2, 3, 4, 5, 6], axis=4),
        dict(shape=[2, 3, 4, 5, 6], axis=-1),
        dict(shape=[2, 3, 4, 5, 6], axis=-2),
        dict(shape=[2, 3, 4, 5, 6], axis=-3),
        dict(shape=[2, 3, 4, 5, 6], axis=-4),
        dict(shape=[2, 3, 4, 5, 6], axis=-5)
    ]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.parametrize("keepdims", [0, 1])
    @pytest.mark.parametrize('op', ['ArgMin', 'ArgMax'])
    @pytest.mark.nightly
    def test_argmin(self, params, keepdims, op, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, op=op, ir_version=ir_version, keepdims=keepdims),
                   ie_device, precision, ir_version, temp_dir=temp_dir)
