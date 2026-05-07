// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/openvino.hpp"
#include "fuzz-utils.h"

static const uint8_t kSplitSequence[] = {
    'F','U','Z','Z','_','N','E','X','T','_','F','I','E','L','D'
};

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* Data, size_t Size) {
    ov::Core core;
    int result = 0;
    try { 
        const auto ir_file_set = create_ir_model_files(Data, Size, kSplitSequence, sizeof(kSplitSequence));
        std::filesystem::path xml_path = std::get<0>(ir_file_set);
        std::filesystem::path bin_path = std::get<1>(ir_file_set);
        ScopedRemove cleanup{xml_path, bin_path};

        auto model = core.read_model(xml_path.string());
        if (model) {
            (void)model->get_name();
            (void)model->outputs();
            (void)model->inputs();
        }
    }
    catch (...) {}

    return result;
}