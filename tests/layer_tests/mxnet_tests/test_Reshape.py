import pytest

from common.layer_test_class import check_ir_version
from common.mxnet_layer_test_class import CommonMXNetLayerTest
from unit_tests.utils.graph import build_graph


class TestReshape(CommonMXNetLayerTest):
    def create_net(self, shape, new_shape, output_shape, ir_version, is_compare = True, reverse=False):
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
        layer_name = 'reshape'
        layer_ptr = mx.symbol.reshape(data=data, name=layer_name, shape=new_shape, reverse=reverse)

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
                'input_data_1': {'shape': [len(output_shape)], 'kind': 'data'},
                'const': {'kind': 'op', 'type': 'Const'},
                'const_data': {'shape': (len(output_shape),), 'kind': 'data'},
                'node': {'kind': 'op', 'type': 'Reshape'},
                'node_data': {'shape': output_shape, 'kind': 'data'},
                'result': {'kind': 'op', 'type': 'Result'}
            }

            ref_net = build_graph(nodes_attributes,
                                  [('input', 'input_data'),
                                   ('input_data_1', 'const'),
                                   ('const', 'const_data'),
                                   ('input_data', 'node'),
                                   ('const_data', 'node'),
                                   ('node', 'node_data'),
                                   ('node_data', 'result')
                                   ])

        if not is_compare:
            ref_net = None

        return mxnet_net, ref_net

    test_data = [dict(shape=[12, 128, 128], new_shape=[-1, 12, 0, 0], output_shape=[1, 12, 128, 128],
                      reverse=True, is_compare=False),
                 dict(shape=[1, 3, 2, 1, 3], new_shape=[1, -2, 3], output_shape=[1, 1, 3, 2, 1, 3],
                      reverse=True, is_compare=False),
                 dict(shape=[2, 4, 5, 5, 1], new_shape=[2, 4, -3, 1], output_shape=[2, 4, 25, 1],
                      reverse=True, is_compare=False),
                 dict(shape=[6, 5, 6], new_shape=[2, 3, -4, 5, 6], output_shape=[2, 3, 5, 6],
                      reverse=True, is_compare=False),
                 dict(shape=[2, 3, 10, 10], new_shape=[-1, 0, 0], output_shape=[6, 10, 10],
                      reverse=True, is_compare=False),
                 dict(shape=[1, 512, 38, 38], new_shape=[1, 512, -2, 1], output_shape=[1, 512, 38, 38, 1]),
                 dict(shape=[1, 512, 2, 3, 4, 5], new_shape=[1, 512, -3, -3, 1], output_shape=[1, 512, 6, 20, 1]),
                 dict(shape=[5, 2, 3, 4, 5, 6, 7, 8], new_shape=[5, -3, -3, -2], output_shape=[5, 6, 20, 6, 7, 8]),
                 dict(shape=[5, 2, 3, 4, 5, 6, 7, 8], new_shape=[5, -2], output_shape=[5, 2, 3, 4, 5, 6, 7, 8],
                      is_compare=False),
                 dict(shape=[2, 3], new_shape=[2, -4, 1, 3, 1], output_shape=[2, 1, 3, 1]),
                 dict(shape=[2, 6, 4], new_shape=[2, -4, -1, 2, 4, 1], output_shape=[2, 3, 2, 4, 1]),
                 dict(shape=[1, 512, 38, 38], new_shape=[1, 512, -1], output_shape=[1, 512, 1444]),
                 dict(shape=[2, 512, 38, 38], new_shape=[2, 512, -1], output_shape=[2, 512, 1444]),
                 dict(shape=[2, 512, 39, 38], new_shape=[1, -1, 512], output_shape=[1, 2964, 512]),
                 dict(shape=[2, 512, 38, 38], new_shape=[1, -1, 512, 2], output_shape=[1, 1444, 512, 2]),
                 dict(shape=[2, 512, 39], new_shape=[-1, 2, 512], output_shape=[39, 2, 512]),
                 dict(shape=[1, 1024], new_shape=[1, 2, 512], output_shape=[1, 2, 512]),
                 dict(shape=[1024], new_shape=[1, 2, 512], output_shape=[1, 2, 512]),
                 dict(shape=[1024], new_shape=[1, -1, 512], output_shape=[1, 2, 512])]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_reshape(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, ir_version=ir_version), ie_device, precision,
                   input_shapes=[params['shape']], ir_version=ir_version, temp_dir=temp_dir)
