import mxnet as mx
import numpy as np
import pytest

from common.layer_test_class import check_ir_version
from common.mxnet_layer_test_class import CommonMXNetLayerTest
from unit_tests.utils.graph import build_graph


class TestScalar(CommonMXNetLayerTest):
    def create_net(self, shape, const, scalar_op, operation, ir_version, **kwargs):
        #
        #   Create MXNet model
        #

        data = mx.symbol.Variable('arg:data')

        if scalar_op == 'mul':
            layer_ptr = data * const
        elif scalar_op == 'sum':
            layer_ptr = data + const
        elif scalar_op == 'div':
            layer_ptr = data / const
        elif scalar_op == 'minus':
            layer_ptr = data - const
        elif scalar_op == 'greater':
            layer_ptr = data > const
        elif scalar_op == 'greater_equal':
            layer_ptr = data >= const
        elif scalar_op == 'equal':
            layer_ptr = data == const
        elif scalar_op == 'not_equal':
            layer_ptr = data != const
        elif scalar_op == 'less':
            layer_ptr = data < const
        elif scalar_op == 'less_equal':
            layer_ptr = data <= const

        elif scalar_op == 'min':
            layer_ptr = mx.symbol.minimum(data, const)

        elif scalar_op == 'max':
            layer_ptr = mx.symbol.maximum(data, const)

        net = layer_ptr.bind(mx.cpu(), args={'arg:data': mx.nd.array(np.random.random_integers(-128, 255, shape),
                                                                     dtype=np.float32)})

        mxnet_net = {'symbol': net._symbol, 'params': net.arg_dict}
        #
        #   Create reference IR net
        #

        ref_net = None

        if check_ir_version(10, None, ir_version):
            if scalar_op in ['mul', 'sum', 'div', 'minus']:
                ref_graph1 = {'input': {'kind': 'op', 'type': 'Parameter'},
                              'input_data': {'shape': shape, 'kind': 'data'},
                              'node': {'kind': 'op', 'type': 'Power', 'power': kwargs['power'],
                                       'scale': kwargs['scale'], 'shift': kwargs['shift']},
                              'node_data': {'shape': shape, 'kind': 'data'},
                              'result': {'kind': 'op', 'type': 'Result'}}
                ref_graph2 = [('input', 'input_data'),
                              ('input_data', 'node'),
                              ('node', 'node_data'),
                              ('node_data', 'result')]
            elif scalar_op in ['greater', 'greater_equal', 'equal', 'not_equal', 'less', 'less_equal', 'max']:
                ref_graph1 = {'input1': {'kind': 'op', 'type': 'Parameter'},
                              'input1_data': {'shape': shape, 'kind': 'data'},
                              'input2': {'kind': 'data'},
                              'input2_const': {'kind': 'op', 'type': 'Const'},
                              'input2_data': {'kind': 'data'},
                              'node': {'kind': 'op', 'type': 'Eltwise', 'operation': operation},
                              'node_data': {'shape': shape, 'kind': 'data'},
                              'result': {'kind': 'op', 'type': 'Result'}}
                ref_graph2 = [('input1', 'input1_data'),
                              ('input2', 'input2_const'),
                              ('input2_const', 'input2_data'),
                              ('input1_data', 'node'),
                              ('input2_data', 'node'),
                              ('node', 'node_data'),
                              ('node_data', 'result')]
            elif scalar_op in ['min']:
                ref_graph1 = {'input1': {'kind': 'op', 'type': 'Parameter'},
                              'input1_data': {'shape': shape, 'kind': 'data'},
                              'input2': {'kind': 'data'},
                              'input2_const': {'kind': 'op', 'type': 'Const'},
                              'input2_data': {'kind': 'data'},
                              'node_power1': {'kind': 'op', 'type': 'Power', 'power': kwargs['power'],
                                              'scale': kwargs['scale'], 'shift': kwargs['shift']},
                              'power1_data': {'shape': shape, 'kind': 'data'},
                              'node': {'kind': 'op', 'type': 'Eltwise', 'operation': operation},
                              'node_data': {'shape': shape, 'kind': 'data'},
                              'node_power': {'kind': 'op', 'type': 'Power', 'power': kwargs['power'],
                                             'scale': kwargs['scale'], 'shift': kwargs['shift']},
                              'power_data': {'shape': shape, 'kind': 'data'},
                              'result': {'kind': 'op', 'type': 'Result'}}

                ref_graph2 = [('input1', 'input1_data'),
                              ('input2', 'input2_const'),
                              ('input2_const', 'input2_data'),

                              ('input1_data', 'node_power1'),
                              ('node_power1', 'power1_data'),
                              ('power1_data', 'node'),

                              ('input2_data', 'node'),

                              ('node', 'node_data'),
                              ('node_data', 'node_power'),
                              ('node_power', 'power_data'),
                              ('node_data', 'result')]

            ref_net = build_graph(ref_graph1, ref_graph2)

        return mxnet_net, ref_net

    test_data = [dict(shape=[1, 120], const=10, scalar_op='mul', operation='mul', power=1, scale=10, shift=0),
                 dict(shape=[1, 120], const=10, scalar_op='sum', operation='sum', power=1, scale=1, shift=10),
                 dict(shape=[1, 120], const=10, scalar_op='div', operation='div', power=1, scale=1 / 10, shift=0),
                 dict(shape=[1, 120], const=10, scalar_op='minus', operation='minus', power=1, scale=1, shift=-10),
                 dict(shape=[1, 120], const=10, scalar_op='greater', operation='greater'),
                 dict(shape=[1, 120], const=10, scalar_op='greater_equal', operation='greater_equal'),
                 dict(shape=[1, 120], const=10, scalar_op='equal', operation='equal'),
                 dict(shape=[1, 120], const=10, scalar_op='not_equal', operation='not_equal'),
                 dict(shape=[1, 120], const=10, scalar_op='less', operation='less'),
                 dict(shape=[1, 120], const=10, scalar_op='less_equal', operation='less_equal'),
                 dict(shape=[1, 120], const=10, scalar_op='min', operation='max', power=1, scale=-1, shift=0),
                 dict(shape=[1, 120], const=10, scalar_op='max', operation='max')]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_eltwise_scalar(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, ir_version=ir_version), ie_device, precision,
                   input_shapes=[params['shape']], input_names=['arg:data'],
                   ir_version=ir_version, temp_dir=temp_dir)

