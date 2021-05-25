import pytest

from common.layer_test_class import check_ir_version
from common.mxnet_layer_test_class import CommonMXNetLayerTest
from unit_tests.utils.graph import build_graph


class TestTranspose(CommonMXNetLayerTest):
    def create_net(self, shape, output_shape, axes, ir_version):
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
        layer_name = 'transpose'
        layer_ptr = mx.symbol.transpose(data=data, axes=axes, name=layer_name)

        net = layer_ptr.bind(mx.cpu(), args={'arg:data': mx.nd.array(np.random.random_integers(1, 255, shape),
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

                'input2': {'kind': 'data'},
                'input2_op': {'kind': 'op'},
                'input2_data': {'kind': 'data'},

                'node': {'kind': 'op', 'type': 'Transpose'},
                'node_data': {'shape': output_shape, 'kind': 'data'},
                'result': {'kind': 'op', 'type': 'Result'}
            }

            ref_net = build_graph(nodes_attributes,
                                  [('input', 'input_data'),

                                   ('input2', 'input2_op'),
                                   ('input2_op', 'input2_data'),
                                   ('input2_data', 'node'),

                                   ('input_data', 'node'),
                                   ('node', 'node_data'),
                                   ('node_data', 'result')
                                   ])

        return mxnet_net, ref_net

    test_data = [dict(shape=[1, 512, 38, 38], output_shape=[1, 38, 512, 38], axes=[0, 2, 1, 3])]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_transpose(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, ir_version=ir_version), ie_device, precision,
                   input_shapes=[params['shape']], ir_version=ir_version, temp_dir=temp_dir)
