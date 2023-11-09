# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import itertools
import os
import warnings
from pathlib import Path

import numpy as np
import openvino as ov

from common.constants import test_device, test_precision
from common.layer_utils import IEInfer, InferAPI20
from common.utils.common_utils import generate_ir_python_api


class CommonLayerTest:
    input_model_key = "input_model"

    def produce_model_path(self, framework_model, save_path):
        raise RuntimeError("This is base class, please implement produce_model_path function for"
                           " the specific framework")

    def get_framework_results(self, inputs_dict, model_path):
        raise RuntimeError("This is base class, please implement get_framework_results function for"
                           " the specific framework")

    def _test(self, framework_model, ref_net, ie_device, precision, ir_version, temp_dir, use_old_api,
              use_new_frontend=True, infer_timeout=60, enabled_transforms='',
              disabled_transforms='', **kwargs):
        """
        :param enabled_transforms/disabled_transforms: string with idxs of transforms that should be enabled/disabled.
                                                       Example: "transform_1,transform_2"
        """
        model_path = self.produce_model_path(framework_model=framework_model, save_path=temp_dir)
        self.use_new_frontend = use_new_frontend
        self.use_old_api = use_old_api
        # TODO Pass environment variables via subprocess environment
        os.environ['MO_ENABLED_TRANSFORMS'] = enabled_transforms
        os.environ['MO_DISABLED_TRANSFORMS'] = disabled_transforms

        compress_to_fp16 = False if precision == 'FP32' else True
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

        if use_new_frontend:
            mo_params["use_new_frontend"] = True
        else:
            mo_params["use_legacy_frontend"] = True

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

        if self.use_old_api:
            ie_engine = IEInfer(model=path_to_xml,
                                weights=path_to_bin,
                                device=ie_device)
        else:
            ie_engine = InferAPI20(model=path_to_xml,
                                   weights=path_to_bin,
                                   device=ie_device,
                                   use_new_frontend=use_new_frontend)
        # Prepare feed dict
        if 'kwargs_to_prepare_input' in kwargs and kwargs['kwargs_to_prepare_input']:
            inputs_dict = self._prepare_input(ie_engine.get_inputs_info(precision),
                                              kwargs['kwargs_to_prepare_input'])
        else:
            inputs_dict = self._prepare_input(ie_engine.get_inputs_info(precision))

        # IE infer:
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

        self._check_inputs_outputs_order(path_to_xml, framework_model)

    def _check_inputs_outputs_order(self, path_to_xml, framework_model):
        core = ov.Core()
        ov_model = core.read_model(path_to_xml)
        ov_inputs = ov_model.inputs
        fw_inputs = self._get_input_names(framework_model)

        # Check number of inputs
        assert len(ov_inputs) == len(fw_inputs), "The number of inputs in original and converted model is different. " \
                                                 "Original model has {} inputs, converted model has {} inputs.".format(
            len(fw_inputs), len(ov_inputs))
        # Check order of inputs
        for idx, fw_name in enumerate(fw_inputs):
            ov_input = ov_inputs[idx]
            assert fw_name in ov_input.names

        # Check number of outputs
        ov_outputs = ov_model.outputs
        fw_outputs = self._get_output_names(framework_model)

        assert len(ov_outputs) == len(
            fw_outputs), "The number of outputs in original and converted model is different. " \
                         "Original model has {} outputs, converted model has {} outputs.".format(len(fw_outputs),
                                                                                                 len(ov_outputs))

        # Check order of outputs
        for idx, fw_name in enumerate(fw_outputs):
            ov_output = ov_outputs[idx]
            assert fw_name in ov_output.names


    # Feed dict for each input is filled with random number.
    # It is possible to redefine this function and generate your own input
    def _prepare_input(self, inputs_dict):
        for input in inputs_dict.keys():
            inputs_dict[input] = np.random.randint(-10, 10, inputs_dict[input]).astype(np.float32)
        return inputs_dict

    def _get_input_names(self, framework_model):
        return []

    def _get_output_names(self, framework_model):
        return []

    def compare_ie_results_with_framework(self, infer_res, framework_res, framework_eps):
        is_ok = True
        from common.utils.common_utils import allclose
        for framework_out_name in framework_res:
            ie_out_name = framework_out_name

            if not allclose(infer_res[ie_out_name], framework_res[framework_out_name],
                            atol=framework_eps,
                            rtol=framework_eps):
                is_ok = False
                print("Max diff is {}".format(
                    np.array(
                        abs(infer_res[ie_out_name] - framework_res[framework_out_name])).max()))
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
