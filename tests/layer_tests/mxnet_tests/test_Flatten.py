import pytest

from common.layer_test_class import check_ir_version
from common.mxnet_layer_test_class import CommonMXNetLayerTest
from unit_tests.utils.graph import build_graph


class TestFlatten(CommonMXNetLayerTest):
    def create_net(self, shape, output_shape, ir_version, **kwargs):
        """
            MXNet net                    IR net

            Input->Flatten->Output   =>    Input->Reshape

        """

        #
        #   Create MXNet model
        #

        import mxnet as mx
        import numpy as np

        data = mx.symbol.Variable('arg:data')
        layer_ptr = mx.symbol.Flatten(data)
        net = layer_ptr.bind(mx.cpu(), args={'arg:data': mx.nd.array(np.random.random_integers(-128, 255, shape),
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
                'data': {'kind': 'data'},
                'data_op': {'kind': 'op'},
                'data_op_data': {'kind': 'data'},
                'node': {'kind': 'op', 'type': 'Reshape'},
                'node_data': {'shape': output_shape, 'kind': 'data'},
                'result': {'kind': 'op', 'type': 'Result'}
            }

            ref_net = build_graph(nodes_attributes,
                                  [('input', 'input_data'),
                                   ('input_data', 'node'),
                                   ('data', 'data_op'),
                                   ('data_op', 'data_op_data'),
                                   ('node', 'node_data'),
                                   ('data_op_data', 'node'),
                                   ('node_data', 'result')
                                   ])

        return mxnet_net, ref_net

    test_data = [dict(shape=[1, 120], output_shape=[1, 120]),
                 dict(shape=[1, 120, 2], output_shape=[1, 240]),
                 dict(shape=[1, 140, 2, 3], output_shape=[1, 840]),
                 dict(shape=[2, 140, 2, 3], output_shape=[2, 840])]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_flatten(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, ir_version=ir_version), ie_device, precision,
                   input_shapes=[params['shape']], ir_version=ir_version, temp_dir=temp_dir)
