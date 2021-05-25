import pytest

from common.layer_test_class import check_ir_version
from common.mxnet_layer_test_class import CommonMXNetLayerTest
from unit_tests.utils.graph import build_graph


class TestDeformablePSROIPooling(CommonMXNetLayerTest):
    def create_net(self, shape, roi_shape, output_shape, pooling_params, ir_version, **kwargs):
        """
            MXNet net                    IR net

            Input->Flatten->Output   =>    Input->Reshape

        """

        #
        #   Create MXNet model
        #
        import mxnet as mx
        import numpy as np

        data = mx.symbol.Variable('data')
        rois = mx.symbol.Variable('rois')
        layer_name = 'deform_pooling'
        layer_ptr = mx.symbol.contrib.DeformablePSROIPooling(data=data, rois=rois, name=layer_name,
                                                             **pooling_params)

        net = layer_ptr.bind(mx.cpu(), args={'data': mx.nd.array(np.random.random_integers(1, 255, shape),
                                                                 dtype=np.float32),
                                             'rois': mx.nd.array(np.random.random_integers(1, 255, roi_shape),
                                                                 dtype=np.float32)
                                             })

        mxnet_net = {'symbol': net._symbol, 'params': net.arg_dict}
        #
        #   Create reference IR net
        #

        ref_net = None

        if check_ir_version(10, None, ir_version):
            nodes_attributes = {
                'input': {'kind': 'op', 'type': 'Parameter'},
                'input_data': {'shape': shape, 'kind': 'data'},

                'input2': {'kind': 'data'},
                'input2_const': {'kind': 'op', 'type': 'Const'},
                'input2_data': {'kind': 'data'},

                'node': {'kind': 'op', 'type': 'DeformablePSROIPooling', 'mode': kwargs['mode'],
                         'pooled_height': kwargs['pooled_height'],
                         'spatial_bins_y': kwargs['spatial_bins_y'], 'spatial_bins_x': kwargs['spatial_bins_x'],
                         'spatial_scale': pooling_params['spatial_scale'], 'part_size': pooling_params['part_size'],
                         'trans_std':  kwargs['trans_std'],
                         'pooled_width': kwargs['pooled_width'], 'output_dim': pooling_params['output_dim'],
                         'group_size': pooling_params['group_size']},
                'node_data': {'shape': output_shape, 'kind': 'data'},
                'result': {'kind': 'op', 'type': 'Result'}
            }

            ref_net = build_graph(nodes_attributes,
                                  [('input', 'input_data'),
                                   ('input2', 'input2_const'),
                                   ('input2_const', 'input2_data'),
                                   ('input_data', 'node'),
                                   ('input2_data', 'node'),
                                   ('node', 'node_data'),
                                   ('node_data', 'result')
                                   ])

        return mxnet_net, ref_net

    test_data = [dict(shape=[1, 1024, 63, 38], roi_shape=[300, 5], output_shape=[300, 882, 3, 3],
                      pooling_params=dict(group_size=3, pooled_size=3, sample_per_part=4, no_trans=True,
                      part_size=3, output_dim=2*441, spatial_scale=0.0625),
                      num_classes=441, mode='bilinear_deformable', pooled_height=3, spatial_bins_y=4,
                      spatial_bins_x=4, trans_std=0, pooled_width=3,)]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_deform_psroipooling(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, ir_version=ir_version), ie_device, precision,
                   input_shapes=[params['shape']], ir_version=ir_version, temp_dir=temp_dir)
