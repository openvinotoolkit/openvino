# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import os
from models_hub_common.test_convert_model import TestConvertModel
from models_hub_common.utils import get_models_list
from openvino import convert_model


def flattenize_tuples(list_input):
    if not isinstance(list_input, (tuple, list)):
        return [list_input]
    unpacked_pt_res = []
    for r in list_input:
        unpacked_pt_res.extend(flattenize_tuples(r))
    return unpacked_pt_res


def flattenize_structure(outputs):
    if not isinstance(outputs, dict):
        outputs = flattenize_tuples(outputs)
        return [i.numpy(force=True) if isinstance(i, torch.Tensor) else i for i in outputs]
    else:
        return dict((k, v.numpy(force=True) if isinstance(v, torch.Tensor) else v) for k, v in outputs.items())


def process_pytest_marks(filepath: str):
    return [
        pytest.param(n, marks=pytest.mark.xfail(reason=r) if m ==
                     "xfail" else pytest.mark.skip(reason=r)) if m else n
        for n, _, m, r in get_models_list(filepath)]


def extract_unsupported_ops_from_exception(e: str) -> list:
    exception_str = "No conversion rule found for operations:"
    for s in e.splitlines():
        it = s.find(exception_str)
        if it >= 0:
            _s = s[it + len(exception_str):]
            ops = _s.replace(" ", "").split(",")
            return ops
    return []


class TestTorchConvertModel(TestConvertModel):
    cached_model = None
    def setup_class(self):
        torch.set_grad_enabled(False)

    def load_model(self, model_name, model_link):
        raise "load_model is not implemented"

    def get_inputs_info(self, model_obj):
        return None

    def prepare_inputs(self, inputs_info):
        inputs = getattr(self, "inputs", self.example)
        if isinstance(inputs, dict):
            return dict((k, v.numpy()) for k, v in inputs.items())
        else:
            return flattenize_structure(inputs)

    def convert_model_impl(self, model_obj):
        if hasattr(self, "mode") and self.mode == "export":
            from torch.fx.experimental.proxy_tensor import make_fx, get_isolated_graphmodule
            from torch.export import export
            from packaging import version
            from openvino.frontend.pytorch.fx_decoder import TorchFXPythonDecoder
            import inspect
            from openvino.frontend.pytorch.utils import prepare_example_inputs_and_model

            input_shapes = []
            input_types = []
            model_obj.eval()
            if isinstance(self.example, dict):
                graph = export(model_obj, tuple(), self.example)
                for input_data in self.example.values():
                    input_types.append(input_data.type())
                    input_shapes.append(input_data.size())
            else:
                graph = export(model_obj, self.example)
                for input_data in self.example:
                    input_types.append(input_data.type())
                    input_shapes.append(input_data.size())
            if version.parse(torch.__version__) >= version.parse("2.2"):
                graph = graph.run_decompositions()

            if isinstance(self.example, dict):
              try:
                gm = get_isolated_graphmodule(graph, tuple(), self.example)
              except:
                gm = get_isolated_graphmodule(graph, tuple(), self.example, tracing_mode='symbolic')
            else:
              try:
                gm = make_fx(graph)(*self.example)
              except:
                gm = make_fx(graph, tracing_mode='symbolic')(*self.example)

            print(gm.code)

            decoder = TorchFXPythonDecoder(gm, gm, input_shapes=input_shapes, input_types=input_types)
            print(list(gm.graph.nodes)[-1].args)
            if isinstance(self.example, dict):
                decoder._input_signature = list(self.example.keys())  
            ov_model = convert_model(decoder, example_input=self.example) 
            if isinstance(self.example, dict):         
                pt_res = model_obj(**self.example)
            else:
                pt_res = model_obj(*self.example)
            if isinstance(pt_res, dict):
                for i, k in enumerate(pt_res.keys()):
                    ov_model.outputs[i].get_tensor().set_names({k})
            ov_model.validate_nodes_and_infer_types()
        else:
            ov_model = convert_model(model_obj,
                                     example_input=self.example,
                                     verbose=True
                                     )
        return ov_model

    def convert_model(self, model_obj):
        try:
            ov_model = self.convert_model_impl(model_obj)
        except Exception as e:
            report_filename = os.environ.get("OP_REPORT_FILE", None)
            if report_filename:
                mode = 'a' if os.path.exists(report_filename) else 'w'
                with open(report_filename, mode) as f:
                    ops = extract_unsupported_ops_from_exception(str(e))
                    if ops:
                        ops = [f"{op} {self.model_name}" for op in ops]
                        f.write("\n".join(ops) + "\n")
            raise e
        return ov_model

    def infer_fw_model(self, model_obj, inputs):
        if isinstance(inputs, dict):
            inps = dict((k, torch.from_numpy(v)) for k, v in inputs.items())
            fw_outputs = model_obj(**inps)
        else:
            fw_outputs = model_obj(*[torch.from_numpy(i) for i in inputs])
        return flattenize_structure(fw_outputs)
