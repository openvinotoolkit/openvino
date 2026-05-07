// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/openvino.hpp"
#include "fuzz-utils.h"

const uint8_t kSplitSequence[] = {'F', 'U', 'Z', 'Z', '_', 'N', 'E', 'X', 'T', '_', 'F', 'I', 'E', 'L', 'D'};

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    ov::Core core;
    int result = 0;
    try {
        std::array<std::tuple<const uint8_t*, size_t>, 2> fuzzing_data = split_data(data, size, kSplitSequence, sizeof(kSplitSequence));

        const uint8_t* net_data = std::get<0>(fuzzing_data[0]);
        size_t net_size = std::get<1>(fuzzing_data[0]);
        const uint8_t* weights_data = std::get<0>(fuzzing_data[1]);
        size_t weights_size = std::get<1>(fuzzing_data[1]);

        std::string net((const char*)net_data, net_size);
        ov::Tensor weights(ov::element::u8, {static_cast<size_t>(weights_size)}, (void*)weights_data);
        auto model = core.read_model(net, weights);
        if (model) {
            (void)model->get_name();
            (void)model->outputs();
            (void)model->inputs();
        }
    } catch (...) {}

    return result;
}
