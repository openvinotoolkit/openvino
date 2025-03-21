# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import numpy as np
from pytorch_layer_test_class import PytorchLayerTest, flattenize_inputs
from copy import deepcopy

class TestRandperm(PytorchLayerTest):
    def _prepare_input(self):
        return ()

    def create_model(self, n):
        class AtenRandperm(torch.nn.Module):
            def __init__(self, n):
                super().__init__()
                self.n = n

            def forward(self):
                return torch.randperm(self.n, dtype=torch.int64)

        return AtenRandperm(n), None, "aten::randperm"

    def is_valid_permutation(self, output, n):
        if hasattr(output, 'detach'):
            arr = output.detach().cpu().numpy().astype(np.int64)
        else:
            arr = np.array(output, dtype=np.int64)
        sorted_arr = np.sort(arr.flatten())
        expected = np.arange(n, dtype=np.int64)
        return np.array_equal(sorted_arr, expected)

    @pytest.mark.parametrize("n", [1, 5, 10])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_randperm_custom(self, n, ie_device, precision, ir_version):
        model, ref_net, op = self.create_model(n)
        inputs = self._prepare_input()
        torch_inputs = [torch.from_numpy(x) if isinstance(x, np.ndarray) else x for x in inputs]
        ov_inputs = flattenize_inputs(inputs)
        trace_model = True
        dynamic_shapes = True
        freeze_model = True

        with torch.no_grad():
            smodel, converted_model = self.convert_directly_via_frontend(
                model, torch_inputs, trace_model, dynamic_shapes, ov_inputs, freeze_model
            )

        from openvino import Core
        core = Core()
        compiled_model = core.compile_model(converted_model, ie_device).
        ov_output_dict = compiled_model(())
        ov_output_tensor = list(ov_output_dict.values())[0]

        assert ov_output_tensor.shape[0] == n, f"Output shape {ov_output_tensor.shape} does not match expected ({n},)"
        assert self.is_valid_permutation(ov_output_tensor, n), (
            f"Output {ov_output_tensor} is not a valid permutation of [0, 1, ..., {n-1}]"
        )

    @pytest.mark.xfail(reason="OpenVINO doesn't support empty tensors for randperm")
    def test_randperm_zero(self, ie_device, precision, ir_version):
        model, ref_net, op = self.create_model(0)
        inputs = self._prepare_input()
        torch_inputs = [torch.from_numpy(x) if isinstance(x, np.ndarray) else x for x in inputs]
        ov_inputs = flattenize_inputs(inputs)
        trace_model = True
        dynamic_shapes = True
        freeze_model = True

        with torch.no_grad():
            smodel, converted_model = self.convert_directly_via_frontend(
                model, torch_inputs, trace_model, dynamic_shapes, ov_inputs, freeze_model
            )
        from openvino import Core
        core = Core()
        compiled_model = core.compile_model(converted_model, ie_device)
        _ = compiled_model(())
