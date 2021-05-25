import pytest

from common.layer_test_class import check_ir_version
from common.mxnet_layer_test_class import CommonMXNetLayerTest
from unit_tests.utils.graph import build_graph


class TestConcat(CommonMXNetLayerTest):
    def create_net(self, shape, dim, input_count, ir_version):
        """
            MXNet net                    IR net

            Input->Concat->Output   =>    Input->Concat

        """

        #
        #   Create MXNet model
        #

        import mxnet as mx
        import numpy as np
        data_inputs = {}
        [data_inputs.update({'arg:data{}'.format(i): mx.nd.array(np.random.random_integers(-128, 255, shape),
                                                                 dtype=np.float32)}) for i in range(input_count)]
        layer_ptr = mx.symbol.concat(*[mx.symbol.Variable('arg:data{}'.format(i)) for i in range(input_count)],
                                     dim=dim)
        net = layer_ptr.bind(mx.cpu(), args=data_inputs)

        mxnet_net = {'symbol': net._symbol, 'params': net.arg_dict}
        #
        #   Create reference IR net
        #
        concat_shape = shape.copy()
        concat_shape[dim] = shape[dim] * input_count

        ref_net = None

        if check_ir_version(10, None, ir_version):
            nodes_attributes = {
                'node': {'kind': 'op', 'type': 'Concat', 'axis': dim},
                'node_data': {'shape': concat_shape, 'kind': 'data'},
                'result': {'kind': 'op', 'type': 'Result'}
            }
            [nodes_attributes.update({'input{}'.format(i): {'kind': 'op', 'type': 'Parameter'},
                                      'input_data{}'.format(i): {'shape': shape, 'kind': 'data'}}) for i in
             range(input_count)]

            edges = [('node', 'node_data'),
                     ('node_data', 'result')]
            [edges.append(('input{}'.format(i), 'input_data{}'.format(i))) for i in range(input_count)]
            [edges.append(('input_data{}'.format(i), 'node')) for i in range(input_count)]

            ref_net = build_graph(nodes_attributes, edges)

        return mxnet_net, ref_net

    test_data = [dict(shape=[1, 120], dim=1, input_count=2),
                 dict(shape=[1, 140, 10], dim=1, input_count=3),
                 dict(shape=[1, 140, 10], dim=2, input_count=2),
                 dict(shape=[1, 140, 10, 20], dim=2, input_count=3)]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_concat(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, ir_version=ir_version), ie_device, precision,
                   input_shapes=[params['shape'] for i in range(params['input_count'])],
                   input_names=['arg:data{}'.format(i) for i in range(params['input_count'])],
                   ir_version=ir_version, temp_dir=temp_dir)
