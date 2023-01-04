# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from openvino.runtime import serialize
from openvino.test_utils import compare_functions
from openvino.tools.mo import convert_model

from common.utils.common_utils import generate_ir


class CommonMOConvertTest:
    @staticmethod
    def generate_ir_python_api(**kwargs):
        output_dir = kwargs['output_dir']
        model_name = kwargs['model_name']
        del kwargs['output_dir']
        model = convert_model(**kwargs)
        serialize(model, str(Path(output_dir, model_name + '.xml')))

    def _test(self, temp_dir, test_params, ref_params):
        """
        Generates two IRs using MO Python API and using cmd tool.
        Then two IRs are compared.
        """
        from openvino.runtime import Core
        core = Core()

        test_params.update({"model_name": 'model_test', "output_dir": temp_dir})
        ref_params.update({"model_name": 'model_ref', "output_dir": temp_dir})

        self.generate_ir_python_api(**test_params)

        exit_code, stderr = generate_ir(**ref_params)
        assert not exit_code, (
            "Reference IR generation failed with {} exit code: {}".format(exit_code, stderr))

        ir_test = core.read_model(Path(temp_dir, 'model_test.xml'))
        ir_ref = core.read_model(Path(temp_dir, 'model_ref.xml'))

        flag, msg = compare_functions(ir_test, ir_ref)
        assert flag, '\n'.join(msg)

    def _test_by_ref_graph(self, temp_dir, test_params, ref_graph, compare_tensor_names=True, compare_layout=True):
        """
        Generates IR using MO Python API, reads it and compares with reference graph.
        """
        from openvino.runtime import Core
        core = Core()

        test_params.update({"model_name": 'model_test', "output_dir": temp_dir})
        self.generate_ir_python_api(**test_params)
        ir_test = core.read_model(Path(temp_dir, 'model_test.xml'))
        flag, msg = compare_functions(ir_test, ref_graph, compare_tensor_names=compare_tensor_names)
        assert flag, msg

        if compare_layout:
            for idx in range(len(ir_test.inputs)):
                assert ir_test.inputs[idx].node.layout == ref_graph.inputs[idx].node.layout
