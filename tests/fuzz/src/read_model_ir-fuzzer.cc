// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/openvino.hpp"
#include "fuzz-utils.h"

constexpr std::string_view kSplitSequence = "FUZZ_NEXT_FIELD";

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    ov::Core core;

    try {
        const auto& [xml_path, bin_path] = create_ir_model_files(data, size, kSplitSequence);
        ScopedRemove cleanup_xml{xml_path};
        ScopedRemove cleanup_bin{bin_path};

        if (const auto model = core.read_model(xml_path); model) {
            model->get_name();
            model->outputs();
            model->inputs();
        }
    }
    catch (...) {}
	return 0;
}