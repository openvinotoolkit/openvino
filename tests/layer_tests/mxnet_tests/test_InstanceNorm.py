import pytest

from common.layer_test_class import check_ir_version
from common.mxnet_layer_test_class import CommonMXNetLayerTest
from unit_tests.utils.graph import build_graph


class TestInstanceNorm(CommonMXNetLayerTest):
    def create_net(self, input_shape, weights_shape, shape_out, ir_version, **kwargs):
        #
        #   Create MXNet model
        #

        import mxnet as mx

        data = mx.symbol.Variable('input_data')

        layer_name = 'InstanceNorm'
        gamma = mx.ndarray.random.normal(-1, 1, weights_shape)
        beta = mx.ndarray.random.normal(-1, 1, weights_shape)
        params = {"arg:{}_gamma".format(layer_name): gamma,
                  "arg:{}_beta".format(layer_name): beta}

        layer_ptr = mx.symbol.InstanceNorm(data, name=layer_name, eps=kwargs['epsilon'])
        mxnet_net = {'symbol': layer_ptr, 'params': params}

        #
        #   Create reference IR net
        #

        ref_net = None

        if check_ir_version(10, None, ir_version):
            nodes_attributes = {
                'input': {'kind': 'op', 'type': 'Parameter'},
                'input_data': {'shape': input_shape, 'kind': 'data'},
                'mvn': {'kind': 'op', 'type': 'MVN', 'eps': kwargs['epsilon'] if kwargs['epsilon'] else 1e-5,
                        'normalize_variance': kwargs['normalize_variance']},
                'mvn_data': {'shape': input_shape, 'kind': 'data'},
                'input_weights_const_data': {'kind': 'data'},
                'weights': {'kind': 'op', 'type': 'Const'},
                'weights_data': {'kind': 'data'},
                'input_biases_const_data': {'kind': 'data'},
                'biases': {'kind': 'op', 'type': 'Const'},
                'biases_data': {'kind': 'data'},
                'node_mul': {'kind': 'op', 'type': 'Multiply'},
                'node_mul_data': {'shape': shape_out, 'kind': 'data'},
                'node_add': {'kind': 'op', 'type': 'Add'},
                'node_add_data': {'shape': shape_out, 'kind': 'data'},
                'result': {'kind': 'op', 'type': 'Result'}
            }

            ref_net = build_graph(nodes_attributes,
                                  [('input', 'input_data'),
                                   ('input_data', 'mvn'),
                                   ('mvn', 'mvn_data'),
                                   ('mvn_data', 'node_mul'),
                                   ('input_weights_const_data', 'weights'),
                                   ('weights', 'weights_data'),
                                   ('weights_data', 'node_mul'),
                                   ('node_mul', 'node_mul_data'),
                                   ('node_mul_data', 'node_add'),
                                   ('input_biases_const_data', 'biases'),
                                   ('biases', 'biases_data'),
                                   ('biases_data', 'node_add'),
                                   ('node_add', 'node_add_data'),
                                   ('node_add_data', 'result')])

        return mxnet_net, ref_net

    test_data = [dict(input_shape=[1, 1, 128], weights_shape=[128], shape_out=(1, 128, 128), epsilon=0.001,
                      normalize_variance=1, mul_op='mul', add_op='sum')]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_instance_norm(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, ir_version=ir_version), ie_device, precision,
                   input_shapes=[params['input_shape']], input_names=['input_data'],
                   ir_version=ir_version, temp_dir=temp_dir)
