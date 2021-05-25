import pytest

from common.layer_test_class import check_ir_version
from common.mxnet_layer_test_class import CommonMXNetLayerTest
from unit_tests.utils.graph import build_graph


class TestLRN(CommonMXNetLayerTest):
    def create_net(self, shape, output_shape, ir_version, **keargs):
        """
            MXNet net                    IR net

            Input->Flatten->Output   =>    Input->Reshape

        """

        #
        #   Create MXNet model
        #
        import mxnet as mx
        import numpy as np

        layer_name = 'lrn'
        data = mx.symbol.Variable('arg:data')
        layer_ptr = mx.symbol.LRN(data, name=layer_name, nsize=keargs['nsize'])
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
                'input2_const': {'kind': 'op', 'type': 'Const'},
                'input2_data': {'kind': 'data'},

                'node': {'kind': 'op', 'type': 'LRN', 'beta': keargs['beta'],
                         'alpha': keargs['alpha']},
                'node_data': {'shape': output_shape, 'kind': 'data'},
                'result': {'kind': 'op', 'type': 'Result'}
            }

            ref_net = build_graph(nodes_attributes,
                                  [('input', 'input_data'),
                                   ('input_data', 'node'),

                                   ('input2', 'input2_const'),
                                   ('input2_const', 'input2_data'),
                                   ('input2_data', 'node'),

                                   ('node', 'node_data'),
                                   ('node_data', 'result')
                                   ])

        return mxnet_net, ref_net

    test_data = [dict(shape=[1, 349], output_shape=[1, 349], nsize=2, region='across',
                      beta=0.75, alpha=0.0001, knorm=2)]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_lrn(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, ir_version=ir_version), ie_device, precision,
                   input_shapes=[params['shape']], ir_version=ir_version, temp_dir=temp_dir)

