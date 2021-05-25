import pytest

from common.layer_test_class import check_ir_version
from common.mxnet_layer_test_class import CommonMXNetLayerTest
from unit_tests.utils.graph import build_graph


class TestSlice(CommonMXNetLayerTest):
    def create_net(self, shape, out_shape, ir_version, **kwargs):
        #
        #   Create MXNet model
        #

        import mxnet as mx
        import numpy as np

        data = mx.symbol.Variable('arg:data')
        layer_ptr = mx.symbol.slice(data, begin=kwargs['begin'], end=kwargs['end'])
        net = layer_ptr.bind(mx.cpu(), args={'arg:data': mx.nd.array(np.random.random_integers(-128, 255, shape),
                                                                     dtype=np.float32)})

        mxnet_net = {'symbol': net._symbol, 'params': net.arg_dict}
        #
        #   Create reference IR net
        #

        ref_net = None

        if check_ir_version(10, None, ir_version):
            begin_mask = [kwargs['begin_mask'][0]-1, kwargs['begin_mask'][0]-1]
            end_mask = [kwargs['end_mask'][0]-1, kwargs['end_mask'][0]-1]
            nodes_attributes = {
                'input': {'kind': 'op', 'type': 'Parameter'},
                'input_data': {'shape': shape, 'kind': 'data'},
                'input_data_1': {'shape': [len(kwargs['begin'])], 'kind': 'data'},
                'const1': {'kind': 'op', 'type': 'Const'},
                'const1_data': {'shape': (len(kwargs['begin']),), 'kind': 'data'},
                'input_data_2': {'shape': [len(kwargs['end'])], 'kind': 'data'},
                'const2': {'kind': 'op', 'type': 'Const'},
                'const2_data': {'shape': (len(kwargs['end']),), 'kind': 'data'},
                'node': {'kind': 'op', 'type': 'StridedSlice', 'begin_mask': begin_mask, 'ellipsis_mask': kwargs['ellipsis_mask'],
                         'end_mask': end_mask, 'new_axis_mask': kwargs['new_axis_mask'], 'shrink_axis_mask': kwargs['shrink_axis_mask']},
                'node_data': {'shape': out_shape, 'kind': 'data'},
                'result': {'kind': 'op', 'type': 'Result'}
            }
            edges = [('input', 'input_data'),
                     ('input_data', 'node'),
                     ('input_data_1', 'const1'),
                     ('const1', 'const1_data'),
                     ('const1_data', 'node'),
                     ('input_data_2', 'const2'),
                     ('const2', 'const2_data'),
                     ('const2_data', 'node'),
                     ('node', 'node_data'),
                     ('node_data', 'result')
                    ]

            ref_net = build_graph(nodes_attributes, edges)

        return mxnet_net, ref_net

    test_data = [dict(shape=[5, 4], out_shape=[2, 1], axis=[0, 1], begin=[1, 1], end=[3, 2],
                      begin_mask=[1, 1], ellipsis_mask=[0, 0], end_mask=[1, 1],
                      new_axis_mask=[0, 0], shrink_axis_mask=[0, 0]),
                 dict(shape=[5, 4], out_shape=[2, 1], axis=[0, 1], begin=[1, 1], end=[3, 2],
                      begin_mask=[1, 1], ellipsis_mask=[0, 0], end_mask=[1, 1],
                      new_axis_mask=[0, 0], shrink_axis_mask=[0, 0], step=[2, 2])]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_slice(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, ir_version=ir_version), ie_device, precision,
                   input_shapes=[params['shape']], input_names=['arg:data'], ir_version=ir_version, temp_dir=temp_dir)
