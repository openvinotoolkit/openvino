import pytest

from common.layer_test_class import check_ir_version
from common.mxnet_layer_test_class import CommonMXNetLayerTest
from unit_tests.utils.graph import build_graph


class TestPlus(CommonMXNetLayerTest):
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
        layer_name = 'pooling'
        layer_ptr = mx.symbol.Pooling(data=data, global_pool=kwargs['global_pool'],
                                      kernel=kwargs['kernel'], stride=kwargs['stride'], pad=kwargs['pad'],
                                      pool_type=kwargs['pool_type'], pooling_convention=kwargs['convention'],
                                      name=layer_name)

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
                'node': {'kind': 'op', 'type': 'AvgPool',
                         'strides': kwargs['stride'], 'kernel': kwargs['kernel'],
                         'rounding_type': kwargs['rounding_type'],
                         'pads_begin': kwargs['pads_begin'], 'pads_end': kwargs['pads_end']},
                'node_data': {'shape': output_shape, 'kind': 'data'},
                'result': {'kind': 'op', 'type': 'Result'}
            }

            ref_net = build_graph(nodes_attributes,
                                  [('input', 'input_data'),
                                   ('input_data', 'node'),
                                   ('node', 'node_data'),
                                   ('node_data', 'result')
                                   ])

        return mxnet_net, ref_net

    test_data = [dict(shape=[1, 512, 38, 38], output_shape=[1, 512, 38, 38],
                      kernel=[1, 1], stride=[1, 1], pad=[0, 0],
                      pool_type='avg', convention='full', global_pool=0, pool_method='avg',
                      rounding_type='ceil', pads_begin=[0, 0], pads_end=[0, 0])]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_plus(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, ir_version=ir_version), ie_device, precision,
                   input_shapes=[params['shape']], ir_version=ir_version, temp_dir=temp_dir)
