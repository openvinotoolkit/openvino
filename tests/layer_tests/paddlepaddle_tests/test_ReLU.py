import pytest

from common.layer_test_class import check_ir_version
from common.pdpd_layer_test_class import CommonPaddlePaddleLayerTest
from unit_tests.utils.graph import build_graph


class TestReLU(CommonPaddlePaddleLayerTest):
    def create_net(self, shape, ir_version):
        """
            PaddlePaddle net           IR net

            Input->ReLU       =>       Input->ReLU

        """

        #
        #   Create PaddlePaddle model
        #

        from paddle import enable_static, fluid, static

        enable_static()

        main_program = static.Program()
        startup_program = static.Program()

        with fluid.program_guard(main_program, startup_program):
            input = fluid.layers.data(name='input', shape=shape, dtype='float32')
            output = fluid.layers.fc(name='output', input=input, size=1, act='relu')

        executor = fluid.Executor(fluid.CPUPlace())
        executor.run(startup_program)

        model_data = {
            "inference_program": main_program,
            "input_layers_names": [input.name],
            "output_layers": [output]
        }
        #
        #   Create reference IR net
        #

        ref_net = None

        if check_ir_version(10, None, ir_version):
            nodes_attributes = {
                'input': {'kind': 'op', 'type': 'Parameter'},
                'input_data': {'shape': shape, 'kind': 'data'},
                'node': {'kind': 'op', 'type': 'ReLU'},
                'node_data': {'shape': shape, 'kind': 'data'},
                'result': {'kind': 'op', 'type': 'Result'}
            }

            ref_net = build_graph(nodes_attributes,
                                  [('input', 'input_data'),
                                   ('input_data', 'node'),
                                   ('node', 'node_data'),
                                   ('node_data', 'result')
                                   ])

        return model_data, ref_net

    test_data = [
        dict(shape=[1]),
        dict(shape=[1, 2]),
        dict(shape=[2, 4, 6]),
        dict(shape=[2, 4, 6, 8]),
    ]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_relu(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, ir_version=ir_version), ie_device, precision, ir_version,
                   temp_dir=temp_dir)
