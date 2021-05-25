import pytest

from common.layer_test_class import check_ir_version
from common.mxnet_layer_test_class import CommonMXNetLayerTest
from unit_tests.utils.graph import build_graph


class TestDeformableConvolution(CommonMXNetLayerTest):
    def create_net(self, shape, output_shape, offset_shape, conv_params, ir_version, **kwargs):
        """
            MXNet net                    IR net

            Input->Conv->Output   =>    Input->Conv

        """

        #
        #   Create MXNet model
        #

        import mxnet as mx

        layer_name = 'def_conv'
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
        offset = mx.symbol.Variable('offset')

        layer_ptr = mx.symbol.contrib.DeformableConvolution(data=data, offset=offset, name=layer_name, **conv_params)

        mxnet_net = {'symbol': layer_ptr, 'params': params}
        #
        #   Create reference IR net
        #

        ref_net = None

        if check_ir_version(10, None, ir_version):
            nodes_attributes = {
                'input': {'kind': 'op', 'type': 'Parameter'},
                'input_data': {'shape': shape, 'kind': 'data'},

                'offset': {'kind': 'op', 'type': 'Parameter'},
                'offset_data': {'shape': offset_shape, 'kind': 'data'},

                'conv_w': {'kind': 'data'},
                'conv_w_op': {'kind': 'op'},
                'conv_w_op_data': {'kind': 'data'},

                'conv': {'kind': 'op', 'type': 'DeformableConvolution',
                         'strides': conv_params['stride'], 'dilations': conv_params['dilate'],
                         'pads_begin': conv_params['pad'], 'pads_end': conv_params['pad'],
                         'group': conv_params['num_group'], 'deformable_group': conv_params['num_deformable_group'],
                         },
                'conv_data': {'shape': output_shape, 'kind': 'data'},
                'result': {'kind': 'op', 'type': 'Result'}
            }
            edges = [('input', 'input_data'),
                     ('offset', 'offset_data'),
                     ('input_data', 'conv'),
                     ('offset_data', 'conv'),

                     ('conv_w', 'conv_w_op'),
                     ('conv_w_op', 'conv_w_op_data'),
                     ('conv_w_op_data', 'conv'),

                     ('conv', 'conv_data'),
                     ('conv_data', 'result')]

            if not conv_params['no_bias']:
                nodes_attributes.update({'conv_b': {'kind': 'data'},
                                         'conv_b_op': {'kind': 'op'},
                                         'conv_b_op_data': {'kind': 'data'},})
                edges.extend([('conv_b', 'conv_b_op'), ('conv_b_op', 'conv_b_op_data'),
                              ('conv_b_op_data', 'conv')])

            ref_net = build_graph(nodes_attributes, edges)

        return mxnet_net, ref_net

    test_data = [dict(shape=[1, 3, 120, 120], offset_shape=[1, 3, 120, 120], output_shape=[1, 32, 120, 120],
                      conv_params={'kernel': [3, 3], 'stride': [1, 1], 'pad': [0, 0],
                                   'dilate': [0, 0], 'num_filter': 32, 'no_bias': False,
                                    'num_group': 1, 'num_deformable_group': 1}),
                 dict(shape=[1, 3, 120, 120], offset_shape=[1, 3, 64, 64], output_shape=[1, 32, 64, 64],
                      conv_params={'kernel': [3, 3], 'stride': [1, 1], 'pad': [0, 0],
                                   'dilate': [0, 0], 'num_filter': 32, 'no_bias': True,
                                   'num_group': 1, 'num_deformable_group': 2}),
                 dict(shape=[1, 96, 120, 120], offset_shape=[1, 96, 64, 64], output_shape=[1, 256, 64, 64],
                      conv_params={'kernel': [5, 5], 'stride': [1, 1], 'pad': [2, 2],
                                   'dilate': [1, 1], 'num_filter': 256, 'no_bias': False,
                                   'num_group': 2, 'num_deformable_group': 2})]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_deformable_conv(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, ir_version=ir_version), ie_device, precision,
                   input_shapes=[params['shape'], params['offset_shape']], input_names=['data', 'offset'],
                   ir_version=ir_version, temp_dir=temp_dir)
