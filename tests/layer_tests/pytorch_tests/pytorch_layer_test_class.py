# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import itertools
import warnings

import numpy as np
from common.constants import test_device, test_precision
from openvino.frontend.pytorch.decoder import TorchScriptPythonDecoder

from openvino.frontend import FrontEndManager
from openvino.runtime import Core, Type, PartialShape


class PytorchLayerTest:
    _type_map = {
        "float64": Type.f64,
        "float32": Type.f32,
        "bool": Type.boolean,
        "int32": Type.i32,
        "int64": Type.i64,
        "int16": Type.i16,
        "int8": Type.i8,
        "uint8": Type.u8
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

    def _test(self, model, ref_net, kind, ie_device, precision, ir_version, infer_timeout=60, dynamic_shapes=True,
              **kwargs):
        """
        :param enabled_transforms/disabled_transforms: string with idxs of transforms that should be enabled/disabled.
                                                       Example: "transform_1,transform_2"
        """
        import torch
        if 'kwargs_to_prepare_input' in kwargs and kwargs['kwargs_to_prepare_input']:
            inputs = self._prepare_input(**kwargs['kwargs_to_prepare_input'])
        else:
            inputs = self._prepare_input()
        with torch.no_grad():
            model.eval()
            if not kwargs.get('trace_model', False):
                model = torch.jit.script(model)
            else:
                torch_inputs = [torch.from_numpy(inp) for inp in inputs]
                model = torch.jit.trace(model, torch_inputs)
            if kwargs.get('freeze_model', True):
                model = torch.jit.freeze(model)
            graph = model.inlined_graph
            print(graph)

            if kind is not None and not isinstance(kind, (tuple, list)):
                kind = [kind]
            if kind is not None:
                for op in kind:
                    assert self._check_kind_exist(graph, op), f"Operation {op} type doesn't exist in provided graph"

            fe_manager = FrontEndManager()
            fe = fe_manager.load_by_framework('pytorch')

            decoder = TorchScriptPythonDecoder(model)

            im = fe.load(decoder)
            om = fe.convert(im)

        torch_inps = [torch.from_numpy(inp) if isinstance(inp, np.ndarray) else inp for inp in inputs]
        
        params = om.get_parameters()
        # todo: support lists and dicts
        for i in range(len(inputs)):
            inp = inputs[i]
            if isinstance(inp, list):
                inputs[i] = np.array(inp)
                if inputs[i].dtype == np.int64:
                    inputs[i] = inputs[i].astype(np.int32)
                inp = inputs[i]
            assert inp.dtype.name in self._type_map, f"Unknown type {inp.dtype}."
            params[i].set_element_type(self._type_map[inp.dtype.name])
            shape = [-1] * len(inp.shape) if dynamic_shapes else inp.shape
            params[i].set_partial_shape(PartialShape(shape))
        om.validate_nodes_and_infer_types()

        # OV infer:
        core = Core()
        compiled = core.compile_model(om, ie_device)
        infer_res = compiled(inputs)

        if hasattr(self, 'skip_framework') and self.skip_framework:
            warnings.warn('Framework is skipped')
            return

        # Framework infer:
        fw_res = model(*torch_inps)

        if not isinstance(fw_res, (tuple)):
            fw_res = (fw_res,)

        output_list = list(infer_res.values())

        flatten_fw_res = []

        def flattenize_list_outputs(res):
            results = []
            for res_item in res:
                # if None is at output we skip it
                if res_item is None:
                    continue
                # If input is list or tuple flattenize it
                if isinstance(res_item, (list, tuple)):
                    decomposed_res = flattenize_list_outputs(res_item)
                    results.extend(decomposed_res)
                    continue
                results.append(res_item)
            return results 
       
        flatten_fw_res = flattenize_list_outputs(fw_res)

        assert len(flatten_fw_res) == len(
            output_list), f'number of outputs not equal, {len(flatten_fw_res)} != {len(output_list)}'
        # check if results dtypes match
        for fw_tensor, ov_tensor in zip(flatten_fw_res, output_list):
            if not isinstance(fw_tensor, torch.Tensor):
                if np.isscalar(fw_tensor):
                    assert fw_tensor == np.array(ov_tensor).item(), f"{fw_tensor} != {np.array(ov_tensor).item()}"
                else:
                    if isinstance(fw_tensor, list):
                        ov_tensor = ov_tensor.tolist()
                        assert ov_tensor == fw_tensor
                    assert type(fw_tensor) == type(ov_tensor)
                continue
            assert torch.tensor(np.array(
                ov_tensor)).dtype == fw_tensor.dtype, f"dtype validation failed: {torch.tensor(np.array(ov_tensor)).dtype} != {fw_tensor.dtype}"

        if 'custom_eps' in kwargs and kwargs['custom_eps'] is not None:
            custom_eps = kwargs['custom_eps']
        else:
            custom_eps = 1e-4

        # Compare Ie results with Framework results
        fw_eps = custom_eps if precision == 'FP32' else 5e-2
        is_ok = True
        for i in range(len(infer_res)):
            cur_fw_res = flatten_fw_res[i].to(memory_format=torch.contiguous_format).numpy(
            ) if isinstance(flatten_fw_res[i], torch.Tensor) else flatten_fw_res[i]
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
