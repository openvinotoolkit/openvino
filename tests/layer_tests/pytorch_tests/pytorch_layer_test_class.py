# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import itertools
import warnings
from copy import deepcopy
import os
import torch
import pytest
import logging
import numpy as np

from common.constants import test_device, test_precision
from openvino.frontend.pytorch.ts_decoder import TorchScriptPythonDecoder
from openvino.frontend import FrontEndManager
from openvino.runtime import Core, Type, PartialShape
import openvino.properties.hint as hints

logging.basicConfig(level=logging.DEBUG)


def skip_check(param):
    return skip_if_export(param) if PytorchLayerTest.use_torch_export() else skip_if_fx(param)


def skip_if_export(param, reason="Unsupported on torch.export"):
    return pytest.param(param, marks=pytest.mark.skipif(PytorchLayerTest.use_torch_export(), reason=reason))


def skip_if_fx(param, reason="Unsupported on torch.fx"):
    return pytest.param(param, marks=pytest.mark.skipif(PytorchLayerTest.use_torch_compile_backend(), reason=reason))


class PytorchLayerTest:
    _type_map = {
        "float64": Type.f64,
        "float32": Type.f32,
        "bool": Type.boolean,
        "int32": Type.i32,
        "int64": Type.i64,
        "int16": Type.i16,
        "int8": Type.i8,
        "uint8": Type.u8,
        "float16": Type.f16
    }

    @staticmethod
    def _check_kind_exist(graph, kind):
        for n in graph.nodes():
            if n.kind() == kind:
                return True
            for b in n.blocks():
                if PytorchLayerTest._check_kind_exist(b, kind):
                    return True
        return False

    @staticmethod
    def use_torch_compile_backend():
        torch_compile_env = os.getenv("PYTORCH_TRACING_MODE")
        if torch_compile_env is not None:
            return torch_compile_env == "TORCHFX"
        return False

    @staticmethod
    def use_torch_export():
        torch_compile_env = os.getenv("PYTORCH_TRACING_MODE")
        if torch_compile_env is not None:
            return torch_compile_env == "EXPORT"
        return False


    def _test(self, model, ref_net, kind, ie_device, precision, ir_version, infer_timeout=60, dynamic_shapes=True,
              **kwargs):
        retries = 0
        max_retries = 3
        last_e = None
        while retries < max_retries:
            try:
                return self._test_impl(model, ref_net, kind, ie_device, precision, ir_version, infer_timeout, dynamic_shapes, **kwargs)
            except RuntimeError as e:
                # This is a potentially sporadic issue
                print(f"An error occurred: {e}. Retrying...")
                last_e = e
                retries += 1
        else:
            raise RuntimeError("Max retries reached. Function execution failed.") from last_e


    def _test_impl(self, model, ref_net, kind, ie_device, precision, ir_version, infer_timeout=60, dynamic_shapes=True,
                   **kwargs):
        """
        :param enabled_transforms/disabled_transforms: string with idxs of transforms that should be enabled/disabled.
                                                       Example: "transform_1,transform_2"
        """
        if 'kwargs_to_prepare_input' in kwargs and kwargs['kwargs_to_prepare_input']:
            inputs = self._prepare_input(**kwargs['kwargs_to_prepare_input'])
        else:
            inputs = self._prepare_input()

        def numpy_to_torch_recursively(x):
            if isinstance(x, tuple):
                return tuple(numpy_to_torch_recursively(y) for y in x)
            elif isinstance(x, dict):
                return dict((k, numpy_to_torch_recursively(y)) for k, y in x.items())
            elif isinstance(x, np.ndarray):
                return torch.from_numpy(x)
            else:
                return x

        torch_inputs = [numpy_to_torch_recursively(inp) for inp in inputs]

        if 'custom_eps' in kwargs and kwargs['custom_eps'] is not None:
            custom_eps = kwargs['custom_eps']
        else:
            custom_eps = 1e-4

        ov_inputs = flattenize_inputs(inputs)

        if self.use_torch_compile_backend():
            self.torch_compile_backend_test(model, torch_inputs, **kwargs)
        else:
            if self.use_torch_export():
                from openvino import convert_model
                from torch.export import export

                em = export(model, tuple(torch_inputs))

                converted_model = convert_model(
                    em, example_input=torch_inputs, verbose=True)
                self._resolve_input_shape_dtype(
                    converted_model, ov_inputs, dynamic_shapes)
                smodel = model
            else:
                trace_model = kwargs.get('trace_model', False)
                freeze_model = kwargs.get('freeze_model', True)
                with torch.no_grad():
                    if kwargs.get('use_convert_model', False):
                        smodel, converted_model = self.convert_via_mo(
                            model, torch_inputs, trace_model, dynamic_shapes, ov_inputs, freeze_model)
                    else:
                        smodel, converted_model = self.convert_directly_via_frontend(
                            model, torch_inputs, trace_model, dynamic_shapes, ov_inputs, freeze_model)
                if kind is not None and not isinstance(kind, (tuple, list)):
                    kind = [kind]
                if kind is not None:
                    for op in kind:
                        assert self._check_kind_exist(
                            smodel.inlined_graph, op), f"Operation {op} type doesn't exist in provided graph"
            # OV infer:
            core = Core()
            config = {}
            if ie_device == "GPU" and precision == "FP32":
                config[hints.inference_precision] = Type.f32
            if "dynamic_quantization_group_size" in kwargs:
                config["DYNAMIC_QUANTIZATION_GROUP_SIZE"] = str(kwargs["dynamic_quantization_group_size"])
            compiled = core.compile_model(converted_model, ie_device, config)
            infer_res = compiled(deepcopy(ov_inputs))

            if hasattr(self, 'skip_framework') and self.skip_framework:
                warnings.warn('Framework is skipped')
                return

            # Framework infer:
            fw_res = smodel(*deepcopy(torch_inputs))

            if not isinstance(fw_res, (tuple)):
                fw_res = (fw_res,)

            output_list = list(infer_res.values())

            flatten_fw_res = []

            flatten_fw_res = flattenize_outputs(fw_res)

            assert len(flatten_fw_res) == len(
                output_list), f'number of outputs not equal, {len(flatten_fw_res)} != {len(output_list)}'
            # check if results dtypes match
            for fw_tensor, ov_tensor in zip(flatten_fw_res, output_list):
                if not isinstance(fw_tensor, torch.Tensor):
                    fw_type = torch.tensor(fw_tensor).numpy().dtype
                    ov_type = ov_tensor.dtype
                    assert ov_type == fw_type, f"dtype validation failed: ov={ov_type} vs fw={fw_type}"
                    continue
                ov_tensor_format = torch.tensor(np.array(ov_tensor))
                assert ov_tensor_format.dtype == fw_tensor.dtype, f"dtype validation failed: ov={ov_tensor_format.dtype} vs fw={fw_tensor.dtype}"
                assert ov_tensor_format.shape == fw_tensor.shape, f"Shapes are not equal: ov={ov_tensor_format.shape} vs fw={fw_tensor.shape}"

            # Compare Ie results with Framework results
            fw_eps = custom_eps if precision == 'FP32' else 5e-2
            is_ok = True
            quantized_ops = False
            if 'quantized_ops' in kwargs and kwargs['quantized_ops'] is not None:
                quantized_ops = kwargs['quantized_ops']
                if quantized_ops:
                    assert 'quant_size' in kwargs, "quant size must be specified for quantized_ops flag"
                    quant_size = kwargs['quant_size']
            for i in range(len(infer_res)):
                cur_fw_res = flatten_fw_res[i].contiguous().numpy(force=True) if isinstance(
                    flatten_fw_res[i], torch.Tensor) else flatten_fw_res[i]
                if np.array(cur_fw_res).size == 0:
                    continue
                cur_ov_res = infer_res[compiled.output(i)]
                n_is_not_close = np.array(cur_fw_res).size - np.isclose(cur_ov_res, cur_fw_res,
                                                                        atol=fw_eps,
                                                                        rtol=fw_eps, equal_nan=True).sum()
                max_diff = np.array(abs(np.array(
                    cur_ov_res, dtype=np.float32) - np.array(cur_fw_res, dtype=np.float32))).max()
                if not quantized_ops and n_is_not_close > 0:
                    is_ok = False
                    print("Max diff is {}".format(max_diff))
                elif quantized_ops and max_diff > np.array(quant_size + fw_eps).max():
                    # To remove sporadic issues, allow any number of error of 1 quant
                    is_ok = False
                    print("Errors outside threshold range: {} with max diff {}, expected at most {} with max diff {}".format(
                        n_is_not_close, max_diff, int(np.log10(cur_fw_res.size)), quant_size + fw_eps))
                else:
                    print("Accuracy validation successful!\n")
                    print("absolute eps: {}, relative eps: {}, errors: {}".format(
                        fw_eps, fw_eps, n_is_not_close))
            assert is_ok, "Accuracy validation failed"

    # Each model should specify inputs
    def _prepare_input(self):
        raise RuntimeError("Please provide inputs generation function")

    def convert_via_mo(self, model, example_input, trace_model, dynamic_shapes, ov_inputs, freeze_model):
        from openvino import convert_model, PartialShape
        if trace_model:
            decoder = TorchScriptPythonDecoder(
                model, example_input=example_input, skip_freeze=not freeze_model)
            kwargs = {"example_input": example_input if len(
                example_input) > 1 or isinstance(example_input[0], dict) else example_input[0]}
        else:
            decoder = TorchScriptPythonDecoder(
                model, skip_freeze=not freeze_model)
            kwargs = {"input": [(i.dtype, PartialShape(
                [-1] * len(i.shape))) for i in example_input]}
        smodel = decoder.pt_module
        print(smodel.inlined_graph)
        if not dynamic_shapes:
            input_shapes = [inp.shape for inp in ov_inputs]
            kwargs["input"] = input_shapes
        om = convert_model(decoder, verbose=True, **kwargs)
        self._resolve_input_shape_dtype(om, ov_inputs, dynamic_shapes)
        return smodel, om

    def convert_directly_via_frontend(self, model, example_input, trace_model, dynamic_shapes, ov_inputs, freeze_model):
        fe_manager = FrontEndManager()
        fe = fe_manager.load_by_framework('pytorch')

        if trace_model:
            decoder = TorchScriptPythonDecoder(
                model, example_input=example_input, skip_freeze=not freeze_model)
        else:
            decoder = TorchScriptPythonDecoder(
                model, skip_freeze=not freeze_model)
        smodel = decoder.pt_module
        print(smodel.inlined_graph)
        im = fe.load(decoder)
        om = fe.convert(im)
        self._resolve_input_shape_dtype(om, ov_inputs, dynamic_shapes)
        return smodel, om

    def _resolve_input_shape_dtype(self, om, ov_inputs, dynamic_shapes):
        params = list(om.inputs)
        for i in range(len(ov_inputs)):
            inp = ov_inputs[i]
            if isinstance(inp, list):
                ov_inputs[i] = np.array(inp)
                if ov_inputs[i].dtype == np.int64:
                    ov_inputs[i] = ov_inputs[i].astype(np.int32)
                inp = ov_inputs[i]
            assert inp.dtype.name in self._type_map, f"Unknown type {inp.dtype}."
            if params[i].get_node().get_element_type().is_dynamic():
                params[i].get_node().set_element_type(
                    self._type_map[inp.dtype.name])
            shape = [-1] * len(inp.shape) if dynamic_shapes else inp.shape
            params[i].get_node().set_partial_shape(PartialShape(shape))
        om.validate_nodes_and_infer_types()
        return om

    def torch_compile_backend_test(self, model, inputs, **kwargs):
        torch._dynamo.reset()
        with torch.no_grad():
            model.eval()
            fw_res = model(*inputs)

        torch._dynamo.reset()
        with torch.no_grad():
            model.eval()
            options={"testing": 1,}
            if ("aot_autograd" in kwargs):
                options.update({"aot_autograd": True,})
            if "dynamic_quantization_group_size" in kwargs:
                options["config"] = {"DYNAMIC_QUANTIZATION_GROUP_SIZE": str(kwargs["dynamic_quantization_group_size"])}
            dynamic = False
            if ("dynamic" in kwargs):
                dynamic = kwargs["dynamic"]

            ov_model = torch.compile(
                model, backend="openvino", dynamic=dynamic, options=options)
            ov_res = ov_model(*inputs)

        if not isinstance(fw_res, (tuple)):
            fw_res = (fw_res,)

        if not isinstance(ov_res, (tuple)):
            ov_res = (ov_res,)

        flatten_fw_res, flatten_ov_res = [], []
        flatten_fw_res = flattenize_outputs(fw_res)
        flatten_ov_res = flattenize_outputs(ov_res)

        assert len(flatten_fw_res) == len(
            flatten_ov_res
        ), f'number of outputs are not equal, {len(flatten_fw_res)} != {len(flatten_ov_res)}'

        # Check if output data types match
        for fw_tensor, ov_tensor in zip(flatten_fw_res, flatten_ov_res):
            if not isinstance(fw_tensor, torch.Tensor) and not isinstance(ov_tensor, torch.Tensor):
                assert fw_tensor == ov_tensor
                assert type(fw_tensor) == type(ov_tensor)
                continue
            assert fw_tensor.dtype == ov_tensor.dtype, f"dtype validation failed: {fw_tensor.dtype} != {ov_tensor.dtype}"

        if 'custom_eps' in kwargs and kwargs['custom_eps'] is not None:
            fw_eps = kwargs['custom_eps']
        else:
            fw_eps = 1e-4
        is_ok = True
        for i in range(len(flatten_ov_res)):
            cur_ov_res = flatten_ov_res[i]
            cur_fw_res = flatten_fw_res[i]
            if not torch.allclose(cur_fw_res, cur_ov_res,
                                  atol=fw_eps, rtol=fw_eps,
                                  equal_nan=True):
                is_ok = False
                print(
                    "Max diff is {}".format(
                        torch.max(torch.tensor(abs(cur_ov_res - cur_fw_res)))
                    )
                )
            else:
                print("Accuracy validation successful!\n")
                print("absolute eps: {}, relative eps: {}".format(fw_eps, fw_eps))
        assert is_ok, "Accuracy validation failed"


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


def flattenize_dict_outputs(res, types):
    if isinstance(res, dict):
        return flattenize(res.values(), types)


def flattenize(res, types: list):
    results = []
    for res_item in res:
        # if None is at output we skip it
        if res_item is None:
            continue
        # If input is list or tuple flattenize it
        if isinstance(res_item, (list, tuple)) and type(res_item) in types:
            decomposed_res = flattenize(res_item, types)
            results.extend(decomposed_res)
            continue
        if isinstance(res_item, dict) and type(res_item) in types:
            decomposed_res = flattenize_dict_outputs(res_item, types)
            results.extend(decomposed_res)
            continue
        results.append(res_item)
    return results


def flattenize_outputs(res):
    return flattenize(res, [list, tuple, dict])


def flattenize_inputs(res):
    return flattenize(res, [tuple, dict])
