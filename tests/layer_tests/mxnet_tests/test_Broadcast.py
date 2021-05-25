import mxnet as mx
import numpy as np
import pytest

from common.layer_test_class import check_ir_version
from common.mxnet_layer_test_class import CommonMXNetLayerTest
from unit_tests.utils.graph import build_graph


class TestBroadcast(CommonMXNetLayerTest):
    def create_net(self, shape, operation, broadcast, expected_op, v10_expected_type, ir_version, **kwargs):
        #
        #   Create MXNet model
        #

        broadcast_structure = {
            # pytest-xdist can't execute the tests in parallel because workers can't compare tests scopes before run
            # mx.symbol.<broadcast> operation have no "==" operation to be compared
            "broadcast_add": mx.symbol.broadcast_add,
            "broadcast_mul": mx.symbol.broadcast_mul,
            "broadcast_div": mx.symbol.broadcast_div,
            "broadcast_sub": mx.symbol.broadcast_sub,
            "minimum": mx.symbol.minimum,
            "maximum": mx.symbol.maximum,
            "broadcast_maximum": mx.symbol.broadcast_maximum,
            "broadcast_minimum": mx.symbol.broadcast_minimum,
            "broadcast_greater": mx.symbol.broadcast_greater,
            "broadcast_greater_equal": mx.symbol.broadcast_greater_equal,
            "broadcast_equal": mx.symbol.broadcast_equal,
            "broadcast_lesser": mx.symbol.broadcast_lesser,
            "broadcast_lesser_equal": mx.symbol.broadcast_lesser_equal,
            "broadcast_power": mx.symbol.broadcast_power,
            "broadcast_not_equal": mx.symbol.broadcast_not_equal,
            "broadcast_logical_and": mx.symbol.broadcast_logical_and,
            "broadcast_logical_or": mx.symbol.broadcast_logical_or,
        }

        broadcast = broadcast_structure[broadcast]

        data1 = mx.symbol.Variable('arg:data1')
        data2 = mx.symbol.Variable('arg:data2')
        layer_ptr = broadcast(data1, data2)
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
            if operation in ['min', 'sub', 'sum', 'mul', 'max', 'greater', 'greater_equal',
                             'equal', 'less', 'less_equal', 'power', 'not_equal', 'logical_and', 'logical_or']:
                ref_graph1 = {'input1': {'kind': 'op', 'type': 'Parameter'},
                              'input1_data': {'shape': shape, 'kind': 'data'},
                              'input2': {'kind': 'op', 'type': 'Parameter'},
                              'input2_data': {'shape': shape, 'kind': 'data'},
                              'node': {'kind': 'op', 'type': v10_expected_type},
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
                              'input2': {'kind': 'op', 'type': 'Parameter'},
                              'input2_data': {'shape': shape, 'kind': 'data'},
                              'input_power': {'kind': 'data', 'type': 'Parameter'},
                              'input_power_op': {'kind': 'op'},
                              'input_power_data': {'kind': 'data'},
                              'node_power': {'kind': 'op', 'type': kwargs['node_power']},
                              'power_data': {'shape': shape, 'kind': 'data'},
                              'node': {'kind': 'op', 'type': v10_expected_type},
                              'node_data': {'shape': shape, 'kind': 'data'},
                              'result': {'kind': 'op', 'type': 'Result'}}

                ref_graph2 = [('input1', 'input1_data'),
                              ('input2', 'input2_data'),
                              ('input_power', 'input_power_op'),
                              ('input_power_op', 'input_power_data'),
                              ('input1_data', 'node_power'),
                              ('input_power_data', 'node_power'),
                              ('node_power', 'power_data'),
                              ('power_data', 'node'),
                              ('input2_data', 'node'),
                              ('node', 'node_data'),
                              ('node_data', 'result')
                              ]

                ref_net = build_graph(ref_graph1, ref_graph2)

        return mxnet_net, ref_net

    test_data = [dict(shape=[1, 120], operation='sum', broadcast="broadcast_add",
                      expected_op='sum', v10_expected_type='Add'),
                 dict(shape=[1, 120], operation='sum', broadcast="broadcast_plus",
                      expected_op='sum', v10_expected_type='Add'),
                 dict(shape=[1, 120], operation='mul', broadcast="broadcast_mul",
                      expected_op='mul', v10_expected_type='Multiply'),
                 dict(shape=[1, 120], operation='div', broadcast="broadcast_div",
                      expected_op='mul', power=-1, scale=1, shift=0, v10_expected_type='Multiply', node_power='Power'),
                 dict(shape=[1, 120], operation='sub', broadcast="broadcast_sub",
                      expected_op='sum', v10_expected_type='Subtract', power=1, scale=-1, shift=0),
                 dict(shape=[1, 120], operation='min', broadcast="minimum",
                      expected_op='max', power=1, scale=-1, shift=0, v10_expected_type='Minimum'),
                 dict(shape=[1, 120], operation='max', broadcast="maximum",
                      expected_op='max', v10_expected_type='Maximum'),
                 dict(shape=[1, 120], operation='max', broadcast="broadcast_maximum",
                      expected_op='max', v10_expected_type='Maximum'),
                 dict(shape=[1, 120], operation='min', broadcast="broadcast_minimum",
                      expected_op='max', power=1, scale=-1, shift=0, v10_expected_type='Minimum'),
                 dict(shape=[1, 120], operation='greater', broadcast="broadcast_greater",
                      expected_op='greater', v10_expected_type='Greater'),
                 dict(shape=[1, 120], operation='greater_equal', broadcast="broadcast_greater_equal",
                      expected_op='greater_equal', v10_expected_type='GreaterEqual'),
                 dict(shape=[1, 120], operation='equal', broadcast="broadcast_equal",
                      expected_op='equal', v10_expected_type='Equal'),
                 dict(shape=[1, 120], operation='less', broadcast="broadcast_lesser",
                      expected_op='less', v10_expected_type='Less'),
                 dict(shape=[1, 120], operation='less_equal', broadcast="broadcast_lesser_equal",
                      expected_op='less_equal', v10_expected_type='LessEqual'),
                 dict(shape=[1, 120], operation='power', broadcast="broadcast_power",
                      expected_op='pow', v10_expected_type='Power'),
                 dict(shape=[1, 120], operation='not_equal', broadcast="broadcast_not_equal",
                      expected_op='not_equal', v10_expected_type='NotEqual'),
                 dict(shape=[1, 120], operation='logical_and', broadcast="broadcast_logical_and",
                      expected_op='logical_and', v10_expected_type='LogicalAnd'),
                 dict(shape=[1, 120], operation='logical_or', broadcast="broadcast_logical_or",
                      expected_op='logical_or', v10_expected_type='LogicalOr')]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_broadcast(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, ir_version=ir_version), ie_device, precision,
                   input_shapes=[params['shape'], params['shape']], input_names=['arg:data1', 'arg:data2'],
                   ir_version=ir_version, temp_dir=temp_dir)
