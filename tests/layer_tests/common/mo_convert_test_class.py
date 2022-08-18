# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from openvino.runtime import serialize
from openvino.tools.mo import convert
from openvino.tools.mo.utils.ir_engine.ir_engine import IREngine

from common.utils.common_utils import generate_ir


class CommonMOConvertTest:
    @staticmethod
    def generate_ir_python_api(**kwargs):
        output_dir = kwargs['output_dir']
        model_name = kwargs['model_name']
        del kwargs['output_dir']
        model = convert(**kwargs)
        serialize(model, str(Path(output_dir, model_name + '.xml')))

    def _test(self, temp_dir, test_params, ref_params):
        """
        Generates two IRs using MO Python API and using cmd tool.
        Then two IRs are compared.
        """
        test_params.update({"model_name": 'model_test', "output_dir": temp_dir})
        ref_params.update({"model_name": 'model_ref', "output_dir": temp_dir})

        self.generate_ir_python_api(**test_params)

        exit_code, stderr = generate_ir(**ref_params)
        assert not exit_code, (
            "Reference IR generation failed with {} exit code: {}".format(exit_code, stderr))

        ir_test = IREngine(Path(temp_dir, 'model_test.xml'), Path(temp_dir, 'model_test.bin'))
        ir_ref = IREngine(Path(temp_dir, 'model_ref.xml'), Path(temp_dir, 'model_ref.bin'))
        flag, resp = ir_test.compare(ir_ref)
        assert flag, '\n'.join(resp)

    def _test_by_ref_graph(self, temp_dir, test_params, ref_graph):
        """
        Generates IR using MO Python API, reads it and compares with reference graph.
        """
        test_params.update({"model_name": 'model_test', "output_dir": temp_dir})

        self.generate_ir_python_api(**test_params)

        ir_test = IREngine(Path(temp_dir, 'model_test.xml'), Path(temp_dir, 'model_test.bin'))
        flag, resp = ir_test.compare(ref_graph)
        assert flag, '\n'.join(resp)
