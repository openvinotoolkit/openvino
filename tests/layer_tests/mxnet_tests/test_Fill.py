import mxnet as mx
import numpy as np
import pytest

from common.layer_test_class import check_ir_version
from common.mxnet_layer_test_class import CommonMXNetLayerTest
from unit_tests.utils.graph import build_graph


class TestFill(CommonMXNetLayerTest):
    def create_net(self, shape, scalar_op, ir_version, **kwargs):
        #
        #   Create MXNet model
        #

        data = mx.symbol.Variable('arg:data')

        if scalar_op == 'zero':
            layer_ptr = mx.symbol.zeros_like(data)
        elif scalar_op == 'one':
            layer_ptr = mx.symbol.ones_like(data)

        net = layer_ptr.bind(mx.cpu(), args={'arg:data': mx.nd.array(np.random.random_integers(-128, 255, shape),
                                                                     dtype=np.float32)})

        mxnet_net = {'symbol': net._symbol, 'params': net.arg_dict}
        #
        #   Create reference IR net
        #

        ref_net = None

        if check_ir_version(10, None, ir_version):
            ref_graph1 = {'input': {'kind': 'op', 'type': 'Parameter'},
                          'input_data': {'shape': shape, 'kind': 'data'},
                          'const': {'kind': 'data'},
                          'const_op': {'kind': 'op'},
                          'const_op_data': {'kind': 'data'},
                          'node': {'kind': 'op', 'type': kwargs['node_type'], },
                          'node_data': {'shape': shape, 'kind': 'data'},
                          'result': {'kind': 'op', 'type': 'Result'}}
            ref_graph2 = [('input', 'input_data'),
                          ('const', 'const_op'),
                          ('const_op', 'const_op_data'),
                          ('const_op_data', 'node'),
                          ('node', 'node_data'),
                          ('node_data', 'result')]
            if scalar_op == 'one':
                ref_graph1.update({'const2': {'kind': 'data'},
                          'const2_op': {'kind': 'op'},
                          'const2_op_data': {'kind': 'data'},
                          'eltwise': {'kind': 'op'},
                          'eltwise_data': {'kind': 'data'},})
                ref_graph2.append(('const2', 'const2_op'))
                ref_graph2.append(('const2_op', 'const2_op_data'))
                ref_graph2.append(('input_data', 'eltwise'))
                ref_graph2.append(('const2_op_data', 'eltwise'))
                ref_graph2.append(('eltwise', 'eltwise_data'))
                ref_graph2.append(('eltwise_data', 'node'))
            else:
                ref_graph2.append(('input_data', 'node'))

            ref_net = build_graph(ref_graph1, ref_graph2)

        return mxnet_net, ref_net

    test_data = [dict(shape=[1, 120], const=10, scalar_op='zero', operation='zero',
                      power=1, scale=0, shift=0, node_type='Multiply'),
                 dict(shape=[1, 120], const=10, scalar_op='one', operation='one',
                      power=1, scale=0, shift=1, node_type='Add')]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_fill(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, ir_version=ir_version), ie_device, precision,
                   input_shapes=[params['shape']], input_names=['arg:data'],
                   ir_version=ir_version, temp_dir=temp_dir)
