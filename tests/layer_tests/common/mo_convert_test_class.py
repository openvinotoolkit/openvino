# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

from common.utils.common_utils import generate_ir
from common.utils.common_utils import shell

from openvino.test_utils import compare_functions
from openvino.tools.ovc import ovc


def generate_ir_ovc(coverage=False, **kwargs):
    # Get OVC file directory
    ovc_path = Path(ovc.__file__).parent

    ovc_runner = ovc_path.joinpath('main.py').as_posix()
    if coverage:
        params = [sys.executable, '-m', 'coverage', 'run', '-p', '--source={}'.format(ovc_runner.parent),
                  '--omit=*_test.py', ovc_runner]
    else:
        params = [sys.executable, ovc_runner]
    for key, value in kwargs.items():
        if key == "input_model":
            params.append((str(value)))
        else:
            params.extend(("--{}".format(key), str(value)))
    exit_code, stdout, stderr = shell(params)
    return exit_code, stderr


class CommonMOConvertTest:
    @staticmethod
    def generate_ir_python_api(**kwargs):
        output_dir = kwargs['output_dir']
        model_name = kwargs['model_name']
        del kwargs['output_dir']
        del kwargs['model_name']
        if 'use_legacy_frontend' in kwargs or 'use_convert_model_from_mo' in kwargs:
            from openvino.tools.mo import convert_model as legacy_convert_model
            from openvino.runtime import serialize
            if 'use_convert_model_from_mo' in kwargs:
                del kwargs['use_convert_model_from_mo']
            model = legacy_convert_model(**kwargs)
            serialize(model, str(Path(output_dir, model_name + '.xml')))
        else:
            from openvino import convert_model, save_model
            # ovc.convert_model does not have 'compress_to_fp16' arg, it's moved into save model
            compress_to_fp16 = True
            if 'compress_to_fp16' in kwargs:
                compress_to_fp16 = kwargs['compress_to_fp16']
                del kwargs['compress_to_fp16']
            model = convert_model(**kwargs)
            save_model(model, str(Path(output_dir, model_name + '.xml')), compress_to_fp16)

    def _test(self, temp_dir, test_params, ref_params):
        """
        Generates two IRs using MO Python API and using cmd tool.
        Then two IRs are compared.
        """
        from openvino import Core
        core = Core()

        test_params.update({"model_name": 'model_test', "output_dir": temp_dir})
        ref_output_path = Path(temp_dir, 'model_ref.xml').absolute().as_posix()
        ref_params.update({"output_model": ref_output_path})

        self.generate_ir_python_api(**test_params)

        exit_code, stderr = generate_ir(**ref_params)
        assert not exit_code, (
            "Reference IR generation failed with {} exit code: {}".format(exit_code, stderr))

        ir_test = core.read_model(Path(temp_dir, 'model_test.xml'))
        ir_ref = core.read_model(Path(temp_dir, 'model_ref.xml'))

        flag, msg = compare_functions(ir_test, ir_ref)
        assert flag, msg

    def _test_by_ref_graph(self, temp_dir, test_params, ref_graph, compare_tensor_names=True,
                           compare_layout=True, ovc=False):
        """
        Generates IR using MO Python API, reads it and compares with reference graph.
        """
        from openvino import Core
        core = Core()

        if ovc:
            ir_file_name = Path(temp_dir, 'model_test.xml')
            test_params.update({"output_model": ir_file_name})
            exit_code, stderr = generate_ir_ovc(coverage=False, **test_params)
            assert not exit_code, stderr
        else:
            test_params.update({"model_name": 'model_test', "output_dir": temp_dir})
            ir_file_name = Path(temp_dir, 'model_test.xml')
            self.generate_ir_python_api(**test_params)

        ir_test = core.read_model(ir_file_name)

        flag, msg = compare_functions(ir_test, ref_graph, compare_tensor_names=compare_tensor_names)
        assert flag, msg

        if compare_layout:
            for idx in range(len(ir_test.inputs)):
                assert ir_test.inputs[idx].node.layout == ref_graph.inputs[idx].node.layout
