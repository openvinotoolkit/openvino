# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class TestMultinomial(PytorchLayerTest):
    def _prepare_input(self, input, num_samples, out):
        model_inputs = [input, np.array(num_samples)]
        if out:
            model_inputs.append(np.zeros_like(input).astype(np.int64))
        return model_inputs

    def create_model(self, replacement, out, test_type):
        class aten_multinomial(torch.nn.Module):
            def __init__(self, replacement, out, test_type) -> None:
                super().__init__()
                self.replacement = replacement
                self.test_type = test_type
                if not out:
                    self.forward = self.multinomial
                else:
                    self.forward = self.multinomial_out

            def multinomial(self, input, num_samples):
                multinomial = torch.multinomial(input, num_samples, self.replacement)
                return self.mode(multinomial)

            def multinomial_out(self, input, num_samples, out):
                multinomial = torch.multinomial(input, num_samples, self.replacement, out=out)
                return self.mode(multinomial)

            def mode(self, op):
                if self.test_type == "shape":
                    return torch.ones_like(op)
                if self.test_type == "sorted":
                    return torch.sort(op)[0]
                return op

        ref_net = None

        return aten_multinomial(replacement, out, test_type), ref_net, "aten::multinomial"

    @pytest.mark.parametrize(
        ("input", "num_samples", "replacement", "test_type"),
        [
            (
                np.array([[0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]], dtype=np.float32),
                1024,
                True,
                "exact",
            ),
            (
                np.array([[0.001, 0.001, 0.1, 0.9], [5, 10, 0.1, 256], [1, 1e-5, 1e-5, 1e-5]], dtype=np.float32),
                4,
                False,
                "sorted",
            ),
            (
                np.array([[0.001, 0, 0.1, 0.9], [5, 10, 0, 256], [0.9, 0.001, 1e-5, 0]], dtype=np.float64),
                3,
                False,
                "sorted",
            ),
            (
                np.array([[1, 2, 3, 4], [4, 3, 2, 1], [1, 0, 0, 0]], dtype=np.float32),
                1,
                True,
                "shape",
            ),
            (
                np.array([1, 2, 3, 4], dtype=np.float32),
                256,
                True,
                "shape",
            ),
            (
                np.array([[1, 2, 3, 4]], dtype=np.float32),
                256,
                True,
                "shape",
            ),
        ],
    )
    @pytest.mark.parametrize("out", [True, False])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_multinomial(self, input, num_samples, replacement, out, test_type, ie_device, precision, ir_version):
        if ie_device == "GPU":
            pytest.xfail(reason="multinomial with num_samples is unsupported on GPU")
        self._test(
            *self.create_model(replacement, out, test_type),
            ie_device,
            precision,
            ir_version,
            kwargs_to_prepare_input={"input": input, "num_samples": num_samples, "out": out}
        )
