// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/openvino.hpp"
#include "fuzz-utils.h"

constexpr std::string_view kSplitSequence = "FUZZ_NEXT_FIELD";

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    ov::Core core;
    try {
        const auto [net_sv, weights_sv] = split_data({reinterpret_cast<const char*>(data), size}, kSplitSequence);

        std::string net{net_sv};
        ov::Tensor weights(ov::element::u8, {weights_sv.size()}, const_cast<char*>(weights_sv.data()));
        if (const auto model = core.read_model(net, weights); model) {
            model->get_name();
            model->outputs();
            model->inputs();
        }
    } catch (...) {}

    return 0;
}
