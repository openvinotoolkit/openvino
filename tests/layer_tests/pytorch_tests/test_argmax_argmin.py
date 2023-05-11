# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestArgMinArgMax(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (np.random.randn(1, 3, 10, 10).astype(np.float32),)

    def create_model(self, op_type, axes, keep_dims):
        import torch
        op_types = {
            'max': torch.argmax,
            'min': torch.argmin
        }

        op = op_types[op_type]

        class aten_argmin_argmax(torch.nn.Module):
            def __init__(self, op):
                super(aten_argmin_argmax, self).__init__()
                self.op = op

            def forward(self, x):
                return self.op(x)

        class aten_argmin_argmax_3arg(torch.nn.Module):
            def __init__(self, op, axes, keep_dims):
                super().__init__()
                self.op = op
                self.axes = axes
                self.keep_dims = keep_dims

            def forward(self, x):
                return self.op(x, self.axes, self.keep_dims)

        ref_net = None
        if axes is None and keep_dims is None:
            model_cls = aten_argmin_argmax(op)
        else:
            model_cls = aten_argmin_argmax_3arg(op, axes, keep_dims)

        return model_cls, ref_net, f"aten::arg{op_type}"

    @pytest.mark.parametrize("axes,keep_dims", [
        (None, None),
        (None, False),
        (None, True),
        (0, False),
        (0, True),
        (1, False),
        (1, True),
        (2, False),
        (2, True),
        (3, False),
        (3, True),
        (-3, False),
        (-3, True),
        (-2, False),
        (-2, True),
        (-1, False),
        (-1, True)])
    @pytest.mark.parametrize("op_type", ['max', 'min'])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_argmin_argmax(self, axes, keep_dims, op_type, ie_device, precision, ir_version):
        self._test(*self.create_model(op_type, axes, keep_dims),
                   ie_device, precision, ir_version, trace_model=True)
