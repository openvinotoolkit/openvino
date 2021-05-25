import pytest

from common.layer_test_class import check_ir_version
from common.mxnet_layer_test_class import CommonMXNetLayerTest
from unit_tests.utils.graph import build_graph


class TestFullyConnected(CommonMXNetLayerTest):
    def create_net(self, shape, output_shape, weights_shape, out_size, no_bias, bias_shape, ir_version):
        """
            MXNet net                    IR net

            Input->Flatten->Output   =>    Input->Reshape

        """

        #
        #   Create MXNet model
        #
        import mxnet as mx

        layer_name = 'fc'
        weights = mx.ndarray.random.normal(-1, 1, weights_shape)
        params = {"arg:{}_weight".format(layer_name): weights}

        if not no_bias:
            bias = mx.ndarray.random.normal(-1, 1, bias_shape)
            params.update({"arg:{}_bias".format(layer_name): bias})

        data = mx.symbol.Variable('data')
        layer_ptr = mx.symbol.FullyConnected(data, name=layer_name, num_hidden=out_size, no_bias=no_bias)

        mxnet_net = {'symbol': layer_ptr, 'params': params}
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
                'input2_op_data': {'kind': 'data'},
                'node1': {'kind': 'op'},
                'node1_data': {'kind': 'data'},
                'input_node2': {'kind': 'data'},
                'input_node2_op': {'kind': 'op'},
                'input_node2_op_data': {'kind': 'data'},
                'node2': {'kind': 'op'},
                'node2_data': {'kind': 'data'},
                'result': {'kind': 'op', 'type': 'Result'}
            }

            edges = [('input', 'input_data'),
                     ('input_data', 'node1'),
                     ('input2', 'input2_op'),
                     ('input2_op', 'input2_op_data'),
                     ('input2_op_data', 'node1'),
                     ('node1', 'node1_data'),
                     ('node1_data', 'node2'),
                     ('input_node2', 'input_node2_op'),
                     ('input_node2_op', 'input_node2_op_data'),
                     ('input_node2_op_data', 'node2'),
                     ('node2', 'node2_data'),
                     ]
            if not no_bias:
                nodes_attributes.update({'input_node3': {'kind': 'data'},
                                         'input_node3_op': {'kind': 'op'},
                                         'input_node3_op_data': {'kind': 'data'},
                                         'node3': {'kind': 'op'},
                                         'node3_data': {'kind': 'data'},
                                         })

                edges.extend([('node2_data', 'node3'), ('input_node3', 'input_node3_op'),
                              ('input_node3_op', 'input_node3_op_data'), ('input_node3_op_data', 'node3'),
                              ('node3', 'node3_data'), ('node3_data', 'result')])
            else:
                edges.extend([('node2_data', 'result')])

            ref_net = build_graph(nodes_attributes, edges)

        return mxnet_net, ref_net

    test_data = [dict(shape=[1, 349], output_shape=[1, 128], out_size=128,
                      weights_shape=(128, 349), no_bias=False, bias_shape=(128,)),
                 dict(shape=[1, 349], output_shape=[1, 128], out_size=128,
                      weights_shape=(128, 349), no_bias=True, bias_shape=(128,)),
                 dict(shape=[1, 9216], output_shape=[1, 128], out_size=128,
                      weights_shape=(128, 9216), no_bias=True, bias_shape=(128,)),
                 dict(shape=[3, 9216], output_shape=[3, 128], out_size=128,
                      weights_shape=(128, 9216), no_bias=False, bias_shape=(128,))]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_fully_connected(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, ir_version=ir_version), ie_device, precision,
                   input_shapes=[params['shape']], ir_version=ir_version, temp_dir=temp_dir)
