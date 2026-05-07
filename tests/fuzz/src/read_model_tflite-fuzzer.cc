// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/openvino.hpp"
#include "fuzz-utils.h"


extern "C" int LLVMFuzzerTestOneInput(const uint8_t* Data, size_t Size) {
    ov::Core core;
    int result = 0;
    try { 
        const std::filesystem::path model_file = create_model_file(Data, Size, ".tflite");
        ScopedRemove cleanup{model_file, {}};
        auto model = core.read_model(model_file.string());
        if (model) {
            (void)model->get_name();
            (void)model->outputs();
            (void)model->inputs();
        }
    } catch (...) {}

    return result;
}
