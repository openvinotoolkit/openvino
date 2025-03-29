# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import openvino as ov
from snippets import get_model


def main():
    model = get_model()
    
    core = ov.Core()
    if "GPU" not in core.available_devices:
        return 0

    #! [dynamic_batch]
    core = ov.Core()

    C = 3
    H = 224
    W = 224

    model.reshape([(1, 10), C, H, W])

    # compile model and create infer request
    compiled_model = core.compile_model(model, "GPU")
    infer_request = compiled_model.create_infer_request()

    # create input tensor with specific batch size
    input_tensor = ov.Tensor(model.input().element_type, [2, C, H, W])

    # ...

    results = infer_request.infer([input_tensor])

    #! [dynamic_batch]

    assert list(results.keys())[0].partial_shape == ov.PartialShape([(1, 10), 3, 224, 224])
    assert list(results.values())[0].shape == tuple(ov.Shape([2, 3, 224, 224]))
