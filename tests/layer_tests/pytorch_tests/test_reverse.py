# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import List
import torch
from openvino.runtime import Core
from openvino import convert_model


def reverse(x, dims: List[int]):
    return torch.flip(x, dims=dims)


def run_reverse_test(dims_list):
    x = torch.randn(1, 3, 4, 5)
    for dims in dims_list:
        print(f"\n🧪 Testing dims = {dims}")
        expected = torch.flip(x, dims=dims)

        # Export to torchscript
        traced = torch.jit.trace(lambda x: reverse(x, dims), (x,))
        traced.save("reverse.pt")

        # Convert using OpenVINO frontend
        core = Core()
        model = convert_model("reverse.pt", example_input=(x,))
        compiled = core.compile_model(model, "CPU")

        # Use Input/Output objects directly
        infer_request = compiled.create_infer_request()
        result = infer_request.infer({compiled.input(0): x.numpy()})
        actual = result[compiled.output(0)]

        assert (actual == expected.numpy()).all(), f"❌ Failed for dims={dims}"
        print(f"✅ Passed for dims = {dims}")


# Run all dims tests
run_reverse_test([[0], [1], [2], [1, 2], [-1], [-2], [0, 2], [0, 1, 2]])


