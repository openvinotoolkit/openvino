import pytest

from common.layer_test_class import check_ir_version
from common.mxnet_layer_test_class import CommonMXNetLayerTest
from unit_tests.utils.graph import build_graph


class TestDeconvolution(CommonMXNetLayerTest):
    def create_net(self, shape, output_shape, conv_params, ir_version):
        """
            MXNet net                    IR net

            Input->Deconv->Output   =>    Input->Deconv

        """

        #
        #   Create MXNet model
        #

        import mxnet as mx

        layer_name = 'deconv'
        num_group = 1

        if 'num_group' in conv_params:
            num_group = conv_params['num_group']
        weights_shape = (conv_params['num_filter'], int(shape[1] / num_group), conv_params['kernel'][0],
                         conv_params['kernel'][1])

        weights = mx.ndarray.random.normal(-1, 1, weights_shape)
        params = {"arg:{}_weight".format(layer_name): weights}

        bias_shape = (conv_params['num_filter'],)
        bias = mx.ndarray.random.normal(-1, 1, bias_shape)
        params.update({"arg:{}_bias".format(layer_name): bias})

        data = mx.symbol.Variable('data')

        layer_ptr = mx.symbol.Deconvolution(data=data, name=layer_name, **conv_params)

        mxnet_net = {'symbol': layer_ptr, 'params': params}
        #
        #   Create reference IR net
        #

        ref_net = None

        if check_ir_version(10, None, ir_version):
            nodes_attributes = {
                'input': {'kind': 'op', 'type': 'Parameter'},
                'input_data': {'shape': shape, 'kind': 'data'},
                'conv_w': {'kind': 'data'},
                'conv_w_op': {'kind': 'op', },
                'conv_w_op_data': {'kind': 'data', },
                'conv_op': {'kind': 'op'},
                'conv_op_data': {'kind': 'data', },
                'result': {'kind': 'op', 'type': 'Result'}
            }
            edges = [('input', 'input_data'),
                     ('input_data', 'conv_op'),
                     ('conv_w', 'conv_w_op'),
                     ('conv_w_op', 'conv_w_op_data'),
                     ('conv_w_op_data', 'conv_op'),
                     ('conv_op', 'conv_op_data')]

            if not conv_params['no_bias']:
                nodes_attributes.update({'conv_b': {'kind': 'data'},
                                         'conv_b_op': {'kind': 'op'},
                                         'conv_b_op_data': {'kind': 'data'},
                                         'add': {'kind': 'op', 'type': 'Add'},
                                         'add_data': {'shape': output_shape, 'kind': 'data'}})

                edges.extend([('conv_b', 'conv_b_op'), ('conv_b_op', 'conv_b_op_data'),
                              ('conv_op_data', 'add'), ('conv_b_op_data', 'add'),
                              ('add', 'add_data'), ('add_data', 'result')])
            else:
                edges.extend([('conv_op_data', 'result')])

            ref_net = build_graph(nodes_attributes, edges)

        return mxnet_net, ref_net

    test_data = [dict(shape=[1, 96, 27, 27], output_shape=[1, 256, 27, 27],
                      conv_params={'kernel': [5, 5], 'stride': [1, 1], 'pad': [2, 2],
                                   'dilate': [1, 1], 'num_filter': 256, 'no_bias': False, 'num_group': 2}),
                 dict(shape=[1, 3, 120, 120], output_shape=[1, 32, 120, 120],
                      conv_params={'kernel': [3, 3], 'stride': [1, 1], 'pad': [0, 0],
                                   'dilate': [0, 0], 'num_filter': 32, 'no_bias': False}),
                 dict(shape=[1, 3, 120, 120], output_shape=[1, 32, 120, 120],
                      conv_params={'kernel': [3, 3], 'stride': [1, 1], 'pad': [0, 0],
                                   'dilate': [0, 0], 'num_filter': 32, 'no_bias': True}),
                 dict(shape=[1, 3, 120, 120], output_shape=[1, 32, 238, 238],
                      conv_params={'kernel': [3, 3], 'stride': [2, 2], 'pad': [1, 1],
                                   'dilate': [0, 0], 'num_filter': 32, 'no_bias': False, 'adj': [1, 1]}),
                 dict(shape=[1, 3, 120, 120], output_shape=[1, 32, 120, 120],
                      conv_params={'kernel': [3, 3], 'stride': [1, 1], 'pad': [0, 0],
                                   'dilate': [0, 0], 'num_filter': 32, 'no_bias': False, 'target_shape': [120, 120]})]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_conv(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, ir_version=ir_version), ie_device, precision,
                   input_shapes=[params['shape']], input_names=['data'], ir_version=ir_version, temp_dir=temp_dir)
