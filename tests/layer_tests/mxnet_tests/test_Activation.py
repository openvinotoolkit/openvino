import pytest

from common.layer_test_class import check_ir_version
from common.mxnet_layer_test_class import CommonMXNetLayerTest
from unit_tests.utils.graph import build_graph


class TestActivation(CommonMXNetLayerTest):
    def create_net(self, shape, act_type, ir_version):
        """
            MXNet net                    IR net

            Input->Activation->Output   =>    Input->Activation(ReLU, Sigmoid, Tanh)

        """

        #
        #   Create MXNet model
        #

        import mxnet as mx
        import numpy as np

        data = mx.symbol.Variable('arg:data')
        layer_ptr = mx.symbol.Activation(data, act_type=act_type.lower())
        net = layer_ptr.bind(mx.cpu(), args={'arg:data': mx.nd.array(np.random.random_integers(-128, 128, shape),
                                                                     dtype=np.float32)})

        mxnet_net = {'symbol': net._symbol, 'params': net.arg_dict}
        #
        #   Create reference IR net
        #
        ref_net = None

        if check_ir_version(10, None, ir_version):
            nodes_attributes = {
                'input': {'kind': 'op', 'type': 'Parameter'},
                'input_data': {'shape': shape, 'kind': 'data'},
                'node': {'kind': 'op', 'type': act_type},
                'node_data': {'shape': shape, 'kind': 'data'},
                'result': {'kind': 'op', 'type': 'Result'}
            }

            ref_net = build_graph(nodes_attributes,
                                  [('input', 'input_data'),
                                   ('input_data', 'node'),
                                   ('node', 'node_data'),
                                   ('node_data', 'result')
                                   ])

        return mxnet_net, ref_net

    test_data = [dict(shape=[1, 120], act_type='ReLU'),
                 dict(shape=[1, 140, 10], act_type='ReLU'),
                 dict(shape=[1, 140, 10, 20], act_type='ReLU'),
                 dict(shape=[1, 140, 10, 20, 30], act_type='ReLU'),

                 dict(shape=[1, 120], act_type='Sigmoid'),
                 dict(shape=[1, 140, 10], act_type='Sigmoid'),
                 dict(shape=[1, 140, 10, 20], act_type='Sigmoid'),
                 dict(shape=[1, 140, 10, 20, 30], act_type='Sigmoid'),

                 dict(shape=[1, 120], act_type='Tanh'),
                 dict(shape=[1, 140, 10], act_type='Tanh'),
                 dict(shape=[1, 140, 10, 20], act_type='Tanh'),
                 dict(shape=[1, 140, 10, 20, 30], act_type='Tanh')
                 ]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_activation(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, ir_version=ir_version),
                   ie_device, precision, input_shapes=[params['shape']], ir_version=ir_version, temp_dir=temp_dir)
