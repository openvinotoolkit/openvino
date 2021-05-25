import mxnet as mx
import numpy as np
import pytest

from common.layer_test_class import check_ir_version
from common.mxnet_layer_test_class import CommonMXNetLayerTest
from unit_tests.utils.graph import build_graph


class TestEltwise(CommonMXNetLayerTest):
    def create_net(self, shape, operation, expected_op, ir_version, **kwargs):
        #
        #   Create MXNet model
        #

        data1 = mx.symbol.Variable('arg:data1')
        data2 = mx.symbol.Variable('arg:data2')

        if operation == 'add':
            layer_ptr = data1 + data2
        elif operation == 'mul':
            layer_ptr = data1 * data2
        elif operation == 'div':
            layer_ptr = data1 / data2
        elif operation == 'sub':
            layer_ptr = data1 - data2

        net = layer_ptr.bind(mx.cpu(), args={'arg:data1': mx.nd.array(np.random.random_integers(-128, 255, shape),
                                                                      dtype=np.float32),
                                             'arg:data2': mx.nd.array(np.random.random_integers(-128, 255, shape),
                                                                      dtype=np.float32)})

        mxnet_net = {'symbol': net._symbol, 'params': net.arg_dict}
        #
        #   Create reference IR net
        #

        ref_net = None

        if check_ir_version(10, None, ir_version):
            if operation in ['add', 'mul', 'sub']:
                ref_graph1 = {'input1': {'kind': 'op', 'type': 'Parameter'},
                              'input1_data': {'shape': shape, 'kind': 'data'},
                              'input2': {'kind': 'op', 'type': 'Parameter'},
                              'input2_data': {'shape': shape, 'kind': 'data'},
                              'node': {'kind': 'op', 'type': kwargs['irv10_type']},
                              'node_data': {'shape': shape, 'kind': 'data'},
                              'result': {'kind': 'op', 'type': 'Result'}}
                ref_graph2 = [('input1', 'input1_data'),
                              ('input2', 'input2_data'),
                              ('input1_data', 'node'),
                              ('input2_data', 'node'),
                              ('node', 'node_data'),
                              ('node_data', 'result')]

            elif operation in ['div']:
                ref_graph1 = {'input1': {'kind': 'op', 'type': 'Parameter'},
                              'input1_data': {'shape': shape, 'kind': 'data'},
                              'input2': {'kind': 'data'},
                              'input2_op': {'kind': 'op'},
                              'input2_data': {'shape': kwargs['input2_data_shape'], 'kind': 'data'},
                              'node_power': {'kind': 'op', 'type': kwargs['irv10_power_type']},
                              'power_data': {'shape': shape, 'kind': 'data'},

                              'input3': {'kind': 'op'},
                              'input3_data': {'kind': 'data'},

                              'node': {'kind': 'op', 'type': kwargs['irv10_type']},
                              'node_data': {'shape': shape, 'kind': 'data'},
                              'result': {'kind': 'op', 'type': 'Result'}}
                ref_graph2 = [('input1', 'input1_data'),
                              ('input2', 'input2_op'),
                              ('input2_op', 'input2_data'),
                              ('input1_data', 'node_power'),
                              ('node_power', 'power_data'),
                              ('power_data', 'node'),
                              ('input2_data', 'node_power'),
                              ('input3', 'input3_data'),
                              ('input3_data', 'node'),
                              ('node', 'node_data'),
                              ('node_data', 'result')]

            ref_net = build_graph(ref_graph1, ref_graph2)

        return mxnet_net, ref_net

    test_data = [dict(shape=[1, 120], operation='mul', expected_op='mul', irv10_type='Multiply'),
                 dict(shape=[1, 120], operation='add', expected_op='sum', irv10_type='Add'),
                 dict(shape=[1, 120], operation='div', expected_op='mul', power=-1, scale=1, shift=0,
                      irv10_type='Multiply', irv10_power_type='Power', input2_data_shape=[1, 1]),
                 dict(shape=[1, 120], operation='sub', expected_op='sum', power=1, scale=-1, shift=0,
                      irv10_type='Subtract')]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_eltwise(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, ir_version=ir_version), ie_device, precision,
                   input_shapes=[params['shape'], params['shape']], input_names=['arg:data1', 'arg:data2'],
                   ir_version=ir_version, temp_dir=temp_dir)

