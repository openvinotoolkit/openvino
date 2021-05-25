import pytest

from common.layer_test_class import check_ir_version
from common.mxnet_layer_test_class import CommonMXNetLayerTest
from unit_tests.utils.graph import build_graph


class TestGather(CommonMXNetLayerTest):
    def create_net(self, shape1, shape2, shape_out, axis, ir_version):
        """
            MXNet net                    IR net

            Input->BroadcastMul->Output   =>    Input->Eltwise

        """

        #
        #   Create MXNet model
        #

        import mxnet as mx
        import numpy as np

        data = mx.symbol.Variable('arg:data1')
        data2 = mx.symbol.Variable('arg:data2')

        layer_ptr = mx.symbol.Embedding(data, data2, input_dim=2000, output_dim=650)
        weights = np.random.random_integers(-128, 255, shape2)
        net = layer_ptr.bind(mx.cpu(), args={'arg:data1': mx.nd.array(np.random.random_integers(-128, 255, shape1),
                                                                      dtype=np.float32),
                                             'arg:data2': mx.nd.array(weights,
                                                                      dtype=np.float32)})

        mxnet_net = {'symbol': net._symbol, 'params': net.arg_dict}
        #
        #   Create reference IR net
        #

        ref_net = None

        if check_ir_version(10, None, ir_version):
            nodes_attributes = {
                'input': {'kind': 'op', 'type': 'Parameter'},
                'input_data': {'shape': shape1, 'kind': 'data'},
                'weights': {'kind': 'op', 'type': 'Input'},
                'weights_data': {'shape': shape1, 'kind': 'data'},
                'node': {'kind': 'op', 'type': 'Gather', 'axis': axis},
                'node_data': {'shape': shape_out, 'kind': 'data'},
                'result': {'kind': 'op', 'type': 'Result'}
            }

            ref_net = build_graph(nodes_attributes,
                                  [('input', 'input_data'),
                                   ('input_data', 'node'),
                                   ('weights', 'weights_data'),
                                   ('weights_data', 'node'),
                                   ('node', 'node_data'),
                                   ('node_data', 'result')])

        return mxnet_net, ref_net

    test_data = [dict(shape1=[35, 32], shape2=[2000, 650], shape_out=(35, 32, 32), axis=0)]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_gather(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, ir_version=ir_version), ie_device, precision,
                   input_shapes=[params['shape1'], params['shape1']], input_names=['arg:data1', 'arg:data2'],
                   ir_version=ir_version, temp_dir=temp_dir)
