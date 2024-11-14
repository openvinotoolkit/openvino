# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import defusedxml.ElementTree as ET
import itertools
import numpy as np
import os
import re
import warnings
from common.constants import test_device, test_precision
from common.layer_utils import InferAPI
from common.utils.common_utils import generate_ir_python_api
from pathlib import Path


class CommonLayerTest:
    input_model_key = "input_model"

    def produce_model_path(self, framework_model, save_path):
        raise RuntimeError("This is base class, please implement produce_model_path function for"
                           " the specific framework")

    def get_framework_results(self, inputs_dict, model_path):
        raise RuntimeError("This is base class, please implement get_framework_results function for"
                           " the specific framework")

    def _test(self, framework_model, ref_net, ie_device, precision, ir_version, temp_dir,
              use_legacy_frontend=False, infer_timeout=60, enabled_transforms='',
              disabled_transforms='', **kwargs):
        """
        :param enabled_transforms/disabled_transforms: string with idxs of transforms that should be enabled/disabled.
                                                       Example: "transform_1,transform_2"
        """
        model_path = self.produce_model_path(framework_model=framework_model, save_path=temp_dir)
        self.use_legacy_frontend = use_legacy_frontend
        # TODO Pass environment variables via subprocess environment
        os.environ['MO_ENABLED_TRANSFORMS'] = enabled_transforms
        os.environ['MO_DISABLED_TRANSFORMS'] = disabled_transforms

        compress_to_fp16 = False if precision == 'FP32' else True

        if use_legacy_frontend:
            mo_params = {self.input_model_key: model_path,
                         "output_dir": temp_dir,
                         "compress_to_fp16": compress_to_fp16,
                         "model_name": 'model'}

            if 'input_shapes' in kwargs and len(kwargs['input_shapes']):
                input_shapes_str = []
                for ishape in kwargs['input_shapes']:
                    input_shapes_str.append('[' + ','.join([str(i) for i in ishape]) + ']')
                mo_params.update(dict(input_shape=','.join(input_shapes_str)))

            if 'input_names' in kwargs and len(kwargs['input_names']):
                mo_params.update(dict(input=','.join(kwargs['input_names'])))
            mo_params["use_legacy_frontend"] = True
        else:
            # pack input parameters for convert_model of OVC
            # that are different from MO
            mo_params = {"input_model": model_path,
                         "output_dir": temp_dir,
                         "compress_to_fp16": compress_to_fp16
                         }

            if 'input_shapes' in kwargs and 'input_names' in kwargs:
                input_shapes = kwargs['input_shapes']
                input_names = kwargs['input_names']
                assert len(input_shapes) == len(input_names)
                input_dict = {}
                for input_name, input_shape in zip(input_names, input_shapes):
                    input_dict[input_name] = input_shape
                mo_params.update(dict(input=input_dict))
            elif 'input_names' in kwargs:
                mo_params.update(dict(input=kwargs['input_names']))
            elif 'input_shapes' in kwargs:
                mo_params.update(dict(input=kwargs['input_shapes']))

        exit_code, stderr = generate_ir_python_api(**mo_params)

        del os.environ['MO_ENABLED_TRANSFORMS']
        del os.environ['MO_DISABLED_TRANSFORMS']
        assert not exit_code, (
            "IR generation failed with {} exit code: {}".format(exit_code, stderr))

        path_to_xml = Path(temp_dir, 'model.xml')
        path_to_bin = Path(temp_dir, 'model.bin')

        # TODO: need to update ref graphs or get rid of this comparison
        # if ref_net is not None:
        #     ir = IREngine(path_to_xml, path_to_bin, precision=precision)
        #     (flag, resp) = ir.compare(ref_net)
        #     assert flag, '\n'.join(resp)

        config = None
        # GPU default execution precision is FP16, so if we want to check FP32 inference
        # we need to set explicit precision hint
        if ie_device == 'GPU' and precision == 'FP32':
            config = {'INFERENCE_PRECISION_HINT': 'f32'}

        ie_engine = InferAPI(model=path_to_xml,
                             weights=path_to_bin,
                             device=ie_device,
                             use_legacy_frontend=use_legacy_frontend)
        # Prepare feed dict
        if 'kwargs_to_prepare_input' in kwargs and kwargs['kwargs_to_prepare_input']:
            inputs_dict = self._prepare_input(ie_engine.get_inputs_info(precision),
                                              kwargs['kwargs_to_prepare_input'])
        else:
            inputs_dict = self._prepare_input(ie_engine.get_inputs_info(precision))

        # OV infer:
        infer_res = ie_engine.infer(input_data=inputs_dict, infer_timeout=infer_timeout, config=config)

        if hasattr(self, 'skip_framework') and self.skip_framework:
            warnings.warn('Framework is skipped')
            return

        # Framework infer:
        fw_res = self.get_framework_results(inputs_dict=inputs_dict, model_path=model_path)

        if 'custom_eps' in kwargs and kwargs['custom_eps'] is not None:
            custom_eps = kwargs['custom_eps']
        else:
            if precision == 'FP32':
                custom_eps = 1e-4
            else:
                custom_eps = 5e-2
        # Compare Ie results with Framework results
        assert self.compare_ie_results_with_framework(infer_res=infer_res, framework_res=fw_res,
                                                      framework_eps=custom_eps), \
            "Comparing with Framework failed: ie_res={}; framework_res={}.".format(infer_res,
                                                                                   fw_res)

        if len(inputs_dict.keys()) > 1 or len(infer_res.keys()) > 1:
            tree = ET.parse(path_to_xml)
            # findall returns elements in document order, this order should be the same as
            # order of inputs/outputs in original model
            inputs_ie = [child for child in tree.findall('.//layer[@type="Parameter"]')]
            outputs_ie = [child for child in tree.findall('.//layer[@type="Result"]')]

            if 'input_names' in kwargs:
                input_names = kwargs['input_names']
                for i, input_name in enumerate(input_names):
                    assert inputs_ie[i].attrib['name'] == input_name, \
                        'Input order does not match framework order. Input with index {} is {}, ' \
                        'but expected {}'.format(i, inputs_ie[i].attrib['name'], input_name)

            if 'output_names' in kwargs:
                output_names = kwargs['output_names']
                for i, output_name in enumerate(output_names):
                    output_name_ie = outputs_ie[i].attrib['name']
                    output_without_sink_port = re.sub(r'\/sink_port_.', '', output_name_ie)

                    assert output_without_sink_port == output_name, \
                        'Output order does not match framework order. Output with index {} is {}, ' \
                        'but expected {}'.format(i, output_without_sink_port, output_name)

    # Feed dict for each input is filled with random number.
    # It is possible to redefine this function and generate your own input
    def _prepare_input(self, inputs_dict):
        for input in inputs_dict.keys():
            inputs_dict[input] = np.random.randint(-10, 10, inputs_dict[input]).astype(np.float32)
        return inputs_dict

    def compare_ie_results_with_framework(self, infer_res, framework_res, framework_eps):
        is_ok = True
        from common.utils.common_utils import allclose
        for framework_out_name in framework_res:
            if framework_out_name not in infer_res and len(infer_res) == 1:
                ie_res = list(infer_res.values())[0]
            else:
                ie_res = infer_res[framework_out_name]

            if not allclose(ie_res, framework_res[framework_out_name],
                            atol=framework_eps,
                            rtol=framework_eps):
                is_ok = False
                if ie_res.dtype != bool:
                    fw_res = np.array(framework_res[framework_out_name])
                    diff = np.array(abs(ie_res - fw_res)).max()
                    print("Max diff is {}".format(diff))
                else:
                    print("Boolean results are not equal")
            else:
                print("Accuracy validation successful!\n")
                print("absolute eps: {}, relative eps: {}".format(framework_eps, framework_eps))
        return is_ok


def get_params(ie_device=None, precision=None):
    """
    :param ie_device: list of devices
    :param precision: list of precisions
    """

    ie_device_params = ie_device if ie_device else test_device
    precision_params = precision if precision else test_precision

    test_args = []
    for element in itertools.product(ie_device_params, precision_params):
        test_args.append(element)
    return test_args


def check_ir_version(left, right, ir_version):
    try:
        _ir_version = int(ir_version)
    except ValueError:
        raise RuntimeError("Wrong ir version type: {}, must be an integer".format(ir_version))
    left_bound = _ir_version - 1 if left is None else left
    right_bound = _ir_version + 1 if right is None else right
    return left_bound <= _ir_version < right_bound
