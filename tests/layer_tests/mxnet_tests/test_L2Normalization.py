import pytest

from common.layer_test_class import check_ir_version
from common.mxnet_layer_test_class import CommonMXNetLayerTest
from unit_tests.utils.graph import build_graph


class TestL2Normalization(CommonMXNetLayerTest):
    def create_net(self, shape, output_shape, weights_shape, no_bias, bias_shape, eps, ir_version):
        """
            MXNet net                    IR net

            Input->Flatten->Output   =>    Input->Reshape

        """

        #
        #   Create MXNet model
        #
        import mxnet as mx
        import numpy as np

        layer_name = 'l2_norm'
        data = mx.symbol.Variable('arg:data')
        layer_ptr = mx.symbol.L2Normalization(data, name=layer_name)
        net = layer_ptr.bind(mx.cpu(), args={'arg:data': mx.nd.array(np.random.random_integers(1, 255, shape),
                                                                     dtype=np.float32)})

        weights = mx.ndarray.random.normal(1, 1, weights_shape)
        params = {"arg:{}_weight".format(layer_name): weights}

        if not no_bias:
            bias = mx.ndarray.random.normal(-1, 1, bias_shape)
            params.update({"arg:{}_bias".format(layer_name): bias})

        mxnet_net = {'symbol': net._symbol, 'params': net.arg_dict}
        #
        #   Create reference IR net
        #

        ref_net = None

        if check_ir_version(10, None, ir_version):
            '''
            nodes_attributes = {
                'input': {'kind': 'op', 'type': 'Parameter'},
                'input_data': {'shape': shape, 'kind': 'data'},
                'node_w': {'kind': 'data'},
                'node': {'kind': 'op', 'type': 'Normalize', 'eps': eps},
                'node_data': {'shape': output_shape, 'kind': 'data'},
                'result': {'kind': 'op', 'type': 'Result'}
            }

            ref_net = build_graph(nodes_attributes,
                                  [('input', 'input_data'),
                                   ('input_data', 'node'),
                                   ('node_w', 'node'),
                                   ('node', 'node_data'),
                                   ('node_data', 'result')
                                   ])
            '''

            nodes_attributes = {
                'input': {'kind': 'op', 'type': 'Parameter'},
                'input_data': {'shape': shape, 'kind': 'data'},
                'node_w': {'kind': 'data'},
                'node_w_op': {'kind': 'op'},
                'node_w_op_data': {'kind': 'data'},
                'node1': {'kind': 'op', 'type': 'NormalizeL2', 'eps': eps},
                'node1_data': {'shape': output_shape, 'kind': 'data'},

                'node2_w': {'kind': 'data'},
                'node2_w_op': {'kind': 'op'},
                'node2_w_op_data': {'kind': 'data'},

                'node2': {'kind': 'op'},
                'node2_data': {'kind': 'data'},

                'result': {'kind': 'op', 'type': 'Result'}
            }

            ref_net = build_graph(nodes_attributes,
                                  [('input', 'input_data'),
                                   ('input_data', 'node1'),
                                   ('node_w', 'node_w_op'),
                                   ('node_w_op', 'node_w_op_data'),
                                   ('node_w_op_data', 'node1'),
                                   ('node1', 'node1_data'),
                                   ('node1_data', 'node2'),

                                   ('node2_w', 'node2_w_op'),
                                   ('node2_w_op', 'node2_w_op_data'),
                                   ('node2_w_op_data', 'node2'),
                                   ('node2', 'node2_data'),
                                   ('node2_data', 'result'),
                                   ])

        return mxnet_net, ref_net

    test_data = [dict(shape=[1, 349], output_shape=[1, 349],
                      weights_shape=(349,), no_bias=False, bias_shape=(349,), eps=1e-10),
                 dict(shape=[1, 349], output_shape=[1, 349],
                      weights_shape=(349,), no_bias=False, bias_shape=(349,), eps=1e-5)]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_l2normalization(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, ir_version=ir_version), ie_device, precision,
                   input_shapes=[params['shape']], ir_version=ir_version, temp_dir=temp_dir)
