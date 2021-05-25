import pytest

from common.layer_test_class import check_ir_version
from common.mxnet_layer_test_class import CommonMXNetLayerTest
from unit_tests.utils.graph import build_graph


class TestMultiBox(CommonMXNetLayerTest):
    def create_net(self, shape, output_shape, ir_version, **kwargs):
        """
            MXNet net                    IR net

            Input->Flatten->Output   =>    Input->Reshape

        """

        #
        #   Create MXNet model
        #
        import mxnet as mx
        import numpy as np

        layer_name = 'prior_box'
        data = mx.symbol.Variable('arg:data')
        layer_ptr = mx.symbol.contrib.MultiBoxPrior(data, name=layer_name,
                                                    sizes=kwargs['sizes'], ratios=kwargs['ratios'],
                                                    clip=kwargs['clip'], steps=kwargs['steps'],
                                                    offsets=kwargs['offsets'])
        net = layer_ptr.bind(mx.cpu(), args={'arg:data': mx.nd.array(np.random.random_integers(1, 255, shape),
                                                                     dtype=np.float32)})

        mxnet_net = {'symbol': net._symbol, 'params': net.arg_dict}
        #
        #   Create reference IR net
        #

        ref_net = None

        if check_ir_version(10, None, ir_version):
            nodes_attributes = {
                'input': {'kind': 'op', 'type': 'Parameter'},
                'input_data': {'shape': shape, 'kind': 'data'},

                'node_left': {'kind': 'op'},
                'node_right': {'kind': 'op'},

                'node_left1_data': {'kind': 'data'},
                'node_left1_op': {'kind': 'op'},
                'node_left1_op_data': {'kind': 'data'},

                'node_left2_data': {'kind': 'data', 'shape': [1]},
                'node_left2_op': {'kind': 'op'},
                'node_left2_op_data': {'kind': 'data', 'shape': [1]},

                'node_left3_data': {'kind': 'data', 'shape': [1]},
                'node_left3_op': {'kind': 'op'},
                'node_left3_op_data': {'kind': 'data', 'shape': [1]},

                'node_left4_data': {'kind': 'data', 'shape': [1]},
                'node_left4_op': {'kind': 'op'},
                'node_left4_op_data': {'kind': 'data', 'shape': [1]},


                'node_right1_data': {'kind': 'data'},
                'node_right1_op': {'kind': 'op'},
                'node_right1_op_data': {'kind': 'data'},

                'node_right2_data': {'kind': 'data', 'shape': [1]},
                'node_right2_op': {'kind': 'op'},
                'node_right2_op_data': {'kind': 'data', 'shape': [1]},

                'node_right3_data': {'kind': 'data', 'shape': [1]},
                'node_right3_op': {'kind': 'op'},
                'node_right3_op_data': {'kind': 'data', 'shape': [1]},

                'node_right4_data': {'kind': 'data', 'shape': [1]},
                'node_right4_op': {'kind': 'op'},
                'node_right4_op_data': {'kind': 'data', 'shape': [1]},

                'node_left1_op2': {'kind': 'op'},
                'node_left1_op2_data': {'kind': 'data'},
                'node_right1_op2': {'kind': 'op'},
                'node_right1_op2_data': {'kind': 'data'},

                'node_left_op3': {'kind': 'op'},
                'node_left_op3_data': {'kind': 'data'},
                'node_right_op3': {'kind': 'op'},
                'node_right_op3_data': {'kind': 'data'},

                'node_op4': {'kind': 'op'},
                'node_op4_data': {'kind': 'data'},

                'node_op5_in': {'kind': 'data'},
                'node_op5_in_op': {'kind': 'op'},
                'node_op5_in_op_data': {'kind': 'data'},

                'node_op5': {'kind': 'op'},
                'node_op5_data': {'kind': 'data'},

                'result': {'kind': 'op', 'type': 'Result'}
            }

            ref_net = build_graph(nodes_attributes,
                                  [('input', 'input_data'),
                                   ('input_data', 'node_left'),
                                   ('input_data', 'node_right'),

                                   ('node_left', 'node_left1_data'),
                                   ('node_left1_data', 'node_left1_op'),
                                   ('node_left1_op', 'node_left1_op_data'),

                                   ('node_left2_data', 'node_left2_op'),
                                   ('node_left2_op', 'node_left2_op_data'),

                                   ('node_left3_data', 'node_left3_op'),
                                   ('node_left3_op', 'node_left3_op_data'),

                                   ('node_left4_data', 'node_left4_op'),
                                   ('node_left4_op', 'node_left4_op_data'),

                                   ('node_left1_op_data', 'node_left1_op2'),
                                   ('node_left2_op_data', 'node_left1_op2'),
                                   ('node_left3_op_data', 'node_left1_op2'),
                                   ('node_left4_op_data', 'node_left1_op2'),

                                   ('node_right', 'node_right1_data'),
                                   ('node_right1_data', 'node_right1_op'),
                                   ('node_right1_op', 'node_right1_op_data'),

                                   ('node_right2_data', 'node_right2_op'),
                                   ('node_right2_op', 'node_right2_op_data'),

                                   ('node_right3_data', 'node_right3_op'),
                                   ('node_right3_op', 'node_right3_op_data'),

                                   ('node_right4_data', 'node_right4_op'),
                                   ('node_right4_op', 'node_right4_op_data'),

                                   ('node_right1_op_data', 'node_right1_op2'),
                                   ('node_right2_op_data', 'node_right1_op2'),
                                   ('node_right3_op_data', 'node_right1_op2'),
                                   ('node_right4_op_data', 'node_right1_op2'),

                                   ('node_left1_op2', 'node_left1_op2_data'),
                                   ('node_right1_op2', 'node_right1_op2_data'),

                                   ('node_left1_op2_data', 'node_left_op3'),
                                   ('node_right1_op2_data', 'node_right_op3'),

                                   ('node_left_op3', 'node_left_op3_data'),
                                   ('node_right_op3', 'node_right_op3_data'),

                                   ('node_left_op3_data', 'node_op4'),
                                   ('node_right_op3_data', 'node_op4'),
                                   ('node_op4', 'node_op4_data'),

                                   ('node_op5_in', 'node_op5_in_op'),
                                   ('node_op5_in_op', 'node_op5_in_op_data'),

                                   ('node_op4_data', 'node_op5'),
                                   ('node_op5_in_op_data', 'node_op5'),
                                   ('node_op5', 'node_op5_data'),
                                   ('node_op5_data', 'result'),
                                   ])

        return mxnet_net, ref_net

    test_data = [dict(shape=[1, 512, 38, 38], output_shape=[1, 2, 5776], sizes=[1], ratios=[1], clip=0,
                      steps=[1, 1], offsets=[1, 1],
                      step_h=0, step_w=0, step=38, scale_all_sizes=0, aspect_ratio=1,
                      img_w=0, img_h=0, flip=0, img_size=0, min_size=38,
                      variance=[0.1, 0.1, 0.2, 0.2])]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_multi_box(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, ir_version=ir_version), ie_device, precision,
                   input_shapes=[params['shape']], ir_version=ir_version, temp_dir=temp_dir)
