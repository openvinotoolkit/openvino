// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/openvino.hpp"
#include "fuzz-utils.h"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    ov::Core core;

    try {
        const auto model_file = create_model_file(data, size, ".onnx");
        ScopedRemove cleanup{model_file};

        if (const auto model = core.read_model(model_file); model) {
            model->get_name();
            model->outputs();
            model->inputs();
        }
    } catch (...) {}
	
    return 0;
}
