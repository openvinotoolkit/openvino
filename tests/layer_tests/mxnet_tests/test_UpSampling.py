import pytest

from common.layer_test_class import check_ir_version
from common.mxnet_layer_test_class import CommonMXNetLayerTest
from unit_tests.utils.graph import build_graph


class TestUpSampling(CommonMXNetLayerTest):
    def create_net(self, shape, output_shape, scale, sample_type, ir_version, num_filter=0, multi_input_mode=None,
                   workspace=512, weight_shape=None, ):
        """
            MXNet net                    IR net

            Input->Flatten->Output   =>    Input->Reshape

        """

        #
        #   Create MXNet model
        #
        import mxnet as mx

        params = {}
        data = mx.symbol.Variable('data')
        layer_name = 'UpSampling'

        if weight_shape:
            weights = mx.ndarray.random.normal(-1, 1, weight_shape)
            params = {"arg:{}_weight".format(layer_name): weights}

        layer_ptr = mx.symbol.UpSampling(data, name=layer_name, scale=scale, sample_type=sample_type,
                                         num_filter=num_filter, multi_input_mode=multi_input_mode, workspace=workspace)

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
                'input2_data': {'kind': 'data'},

                'node': {'kind': 'op', },
                'node_data': {'shape': output_shape, 'kind': 'data'},
                'result': {'kind': 'op', 'type': 'Result'}
            }

            edges = [('input', 'input_data'),
                     ('input_data', 'node'),

                     ('input2', 'input2_op'),
                     ('input2_op', 'input2_data'),
                     ('input2_data', 'node'),

                     ('node', 'node_data'),
                     ('node_data', 'result')]

            if weight_shape:
                nodes_attributes['node']['type'] = 'GroupConvolutionBackpropData'
            else:
                nodes_attributes['node']['type'] = 'Interpolate'

            ref_net = build_graph(nodes_attributes, edges)

        return mxnet_net, ref_net

    test_data = [dict(shape=[1, 512, 38, 38], output_shape=[1, 512, 38*2, 38*2], scale=2,
                      sample_type='nearest', multi_input_mode='concat'),
                 dict(shape=[1, 128, 6, 6], output_shape=[1, 128, 6 * 2, 6 * 2], scale=2,
                      num_filter=128, sample_type='bilinear', multi_input_mode='concat',
                      workspace=512, weight_shape=[128, 1, 4, 4])]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_upsampling(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, ir_version=ir_version), ie_device, precision,
                   input_shapes=[params['shape']], ir_version=ir_version, temp_dir=temp_dir)
