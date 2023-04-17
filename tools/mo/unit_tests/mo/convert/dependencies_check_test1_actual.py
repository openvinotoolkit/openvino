# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

from openvino.tools.mo.utils.error import FrameworkError


def simple_pytorch_model():
    from torch import nn

    class NeuralNetwork(nn.Module):
        def __init__(self):
            super(NeuralNetwork, self).__init__()
            self.linear_relu_stack = nn.Sequential(
                nn.ReLU(),
                nn.Sigmoid(),
            )

        def forward(self, x):
            logits = self.linear_relu_stack(x)
            return logits

    return NeuralNetwork()


def mocked_check_module_import(module_name, required_version, key, not_satisfied_versions):
    if module_name == 'openvino-telemetry':
        raise ImportError()


# Patch check_module_import to have unsatisfied dependency
@patch('openvino.tools.mo.utils.versions_checker.check_module_import', mocked_check_module_import)
@patch('openvino.tools.mo.convert_impl.moc_emit_ir', side_effect=FrameworkError('FW ERROR MESSAGE'))
def run_main(mocked_check_module_import):
    from openvino.tools.mo import convert_model

    # convert_model() should fail to convert and show unsatisfied dependency
    convert_model(simple_pytorch_model(), silent=False)


if __name__ == "__main__":
    run_main()
