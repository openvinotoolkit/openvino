# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import itertools
import jax
import numpy as np
import os
from common.constants import test_device, test_precision
from copy import deepcopy
from openvino.runtime import Core


class JaxLayerTest:

    @staticmethod
    def _check_kind_exist(jaxpr, kind):
        for node in jaxpr.eqns:
            if node.primitive.name == kind:
                return True
        return False

    @staticmethod
    def use_jaxpr_tracing():
        jax_compile_env = os.getenv("JAX_TRACE_MODE")
        if jax_compile_env is not None:
            return jax_compile_env == "JAXPR"
        return False

    def _test(self, model, ref_net, kind, ie_device, precision, ir_version, infer_timeout=60, dynamic_shapes=True,
              **kwargs):
        """
        :param enabled_transforms/disabled_transforms: string with idxs of transforms that should be enabled/disabled.
                                                       Example: "transform_1,transform_2"
        """
        inputs = self._prepare_input()
        if self.use_jaxpr_tracing():
            converted_model, jaxpr = self.convert_via_jaxpr(model, inputs)
            # check that expected operation presents in a graph
            if kind is not None:
                assert self._check_kind_exist(jaxpr, kind), '{} operation does not present in a graph'.format(kind)
        else:
            converted_model = self.convert_via_tensorflow_function(model, inputs)

        # OV infer:
        core = Core()
        compiled = core.compile_model(converted_model, ie_device)
        infer_res = compiled(deepcopy(inputs))

        # Framework infer:
        fw_res = model(*deepcopy(inputs))

        if not isinstance(fw_res, (tuple)):
            fw_res = (fw_res,)

        output_list = ov_res_to_list(infer_res)

        def flattenize_dict_outputs(res):
            if isinstance(res, dict):
                return flattenize_outputs(res.values())

        def flattenize_outputs(res):
            results = []
            for res_item in res:
                # if None is at output we skip it
                if res_item is None:
                    continue
                # If input is list or tuple flattenize it
                if isinstance(res_item, (list, tuple)):
                    decomposed_res = flattenize_outputs(res_item)
                    results.extend(decomposed_res)
                    continue
                if isinstance(res_item, dict):
                    decomposed_res = flattenize_dict_outputs(res_item)
                    results.extend(decomposed_res)
                    continue
                results.append(res_item)
            return results

        flatten_fw_res = flattenize_outputs(fw_res)

        assert len(flatten_fw_res) == len(
            output_list), f'number of outputs not equal, {len(flatten_fw_res)} != {len(output_list)}'
        # check if results dtypes match
        for fw_tensor, ov_tensor in zip(flatten_fw_res, output_list):
            fw_tensor_type = np.array(fw_tensor).dtype
            ov_tensor_type = ov_tensor.dtype
            assert ov_tensor_type == fw_tensor_type, f"dtype validation failed: {ov_tensor_type} != {fw_tensor_type}"

        if 'custom_eps' in kwargs and kwargs['custom_eps'] is not None:
            custom_eps = kwargs['custom_eps']
        else:
            custom_eps = 1e-4

        # compare OpenVINO results with JAX results
        fw_eps = custom_eps if precision == 'FP32' else 5e-2
        is_ok = True
        for i in range(len(flatten_fw_res)):
            cur_fw_res = np.array(flatten_fw_res[i])
            cur_ov_res = infer_res[compiled.output(i)]
            print(f"fw_re: {cur_fw_res};\n ov_res: {cur_ov_res}")
            if not np.allclose(cur_ov_res, cur_fw_res,
                               atol=fw_eps,
                               rtol=fw_eps, equal_nan=True):
                is_ok = False
                print("Max diff is {}".format(
                    np.array(
                        abs(cur_ov_res - cur_fw_res)).max()))
            else:
                print("Accuracy validation successful!\n")
                print("absolute eps: {}, relative eps: {}".format(fw_eps, fw_eps))
        assert is_ok, "Accuracy validation failed"

    # Each model should specify inputs
    def _prepare_input(self):
        raise RuntimeError("Please provide inputs generation function")

    def convert_via_tensorflow_function(self, model, inputs):
        import tensorflow as tf
        from jax.experimental import jax2tf
        from openvino.tools.ovc import convert_model
        # create function signature based on input shapes and types
        function_signature = []
        for _input in inputs:
            if isinstance(_input, jax.Array):
                _input = np.array(_input)
            assert isinstance(_input, np.ndarray)
            input_shape = _input.shape
            input_type = _input.dtype
            function_signature.append(tf.TensorSpec(input_shape, input_type))

        f = tf.function(jax2tf.convert(model,
                                       # request JAX to be lowered with one TensorFlow op for each JAX primitive
                                       # avoid JAX compilation into StableHLO representation wrapped in XlaCallModule
                                       # in early JAX version native_serialization was default but not now
                                       native_serialization=False),
                        autograph=False, input_signature=function_signature)
        converted_model = convert_model(f)
        return converted_model

    def convert_via_jaxpr(self, model, inputs):
        import jax
        from openvino.tools.ovc import convert_model

        jaxpr = jax.make_jaxpr(model)(*inputs)
        converted_model = convert_model(jaxpr)

        return converted_model, jaxpr


def get_params(ie_device=None, precision=None):
    """
    :param ie_device: list of devices
    :param precision: list of precisions
    """

    ie_device_params = ie_device if ie_device else test_device
    precision_params = precision if precision else test_precision

    test_args = []
    for element in itertools.product(ie_device_params, precision_params):
        if element[0] == 'CPU' and element[1] == 'FP16':
            continue
        test_args.append(element)
    return test_args


def ov_res_to_list(ov_res_dict):
    # 118221: remove this WA that clean-up repeating output tensors
    # with the same tensor names
    # probably, we do not utilize some meta info from tf.function
    values = []
    met_names = set()
    for ov_res_name, ov_res_value in ov_res_dict.items():
        if bool(set(ov_res_name.names) & met_names):
            continue
        met_names |= set(ov_res_name.names)
        values.append(ov_res_value)
    return values
