# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import torch
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
        raise RuntimeError("load_model is not implemented")

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
            from torch.export import export
            from packaging import version

            model_obj.eval()
            graph = None
            export_kwargs = {}
            if getattr(self, "export_kwargs", None):
                export_kwargs = self.export_kwargs
            if isinstance(self.example, dict):
                pt_res = model_obj(**self.example)
                graph = export(model_obj, args=tuple(), kwargs=self.example, **export_kwargs)
            else:
                pt_res = model_obj(*self.example)
                graph = export(model_obj, self.example, **export_kwargs)
            ov_model = convert_model(graph, verbose=True)

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
