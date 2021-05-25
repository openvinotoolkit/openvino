import pytest

from common.layer_test_class import check_ir_version
from common.mxnet_layer_test_class import CommonMXNetLayerTest
from unit_tests.utils.graph import build_graph


class TestSliceAxis(CommonMXNetLayerTest):
    def create_net(self, shape, out_shape, axis, begin, end, ir_version):
        #
        #   Create MXNet model
        #

        import mxnet as mx
        import numpy as np

        data = mx.symbol.Variable('arg:data')
        layer_ptr = mx.symbol.slice_axis(data, axis=axis, begin=begin, end=end)
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

                'input2': {'kind': 'data'},
                'input2_op': {'kind': 'op'},
                'input2_data': {'kind': 'data'},

                'input3': {'kind': 'data'},
                'input3_op': {'kind': 'op'},
                'input3_data': {'kind': 'data'},

                'input4': {'kind': 'data'},
                'input4_op': {'kind': 'op'},
                'input4_data': {'kind': 'data'},

                'node': {'kind': 'op', 'type': 'StridedSlice'},
                'node_data': {'shape': out_shape, 'kind': 'data'},
                'result': {'kind': 'op', 'type': 'Result'}
            }

            ref_net = build_graph(nodes_attributes,
                                  [('input', 'input_data'),
                                   ('input_data', 'node'),

                                   ('input2', 'input2_op'),
                                   ('input2_op', 'input2_data'),
                                   ('input2_data', 'node'),

                                   ('input3', 'input3_op'),
                                   ('input3_op', 'input3_data'),
                                   ('input3_data', 'node'),

                                   ('input4', 'input4_op'),
                                   ('input4_op', 'input4_data'),
                                   ('input4_data', 'node'),

                                   ('node', 'node_data'),
                                   ('node_data', 'result')
                                   ])

        return mxnet_net, ref_net

    test_data = [dict(shape=[3, 4], out_shape=[2, 4], axis=0, begin=1, end=3)]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_slice_axis(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, ir_version=ir_version), ie_device, precision,
                   input_shapes=[params['shape']], input_names=['arg:data'], ir_version=ir_version, temp_dir=temp_dir)
