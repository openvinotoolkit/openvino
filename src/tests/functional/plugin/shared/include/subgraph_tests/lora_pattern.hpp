// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/lora_pattern.hpp"

namespace ov {
namespace test {

TEST_P(LoraPatternMatmul, empty_tensors) {
    size_t M = std::get<2>(GetParam());
    size_t N = std::get<3>(GetParam());
    size_t K = std::get<4>(GetParam());

    targetStaticShapes = {{{{1, M, K}}, {{N, K}}}};
    run_test_empty_tensors();
}

TEST_P(LoraPatternConvolution, empty_tensors) {
    size_t num_channels = std::get<2>(GetParam());

    targetStaticShapes = {{{1, num_channels, 64, 64}}};
    run_test_empty_tensors();
}

TEST_P(LoraPatternMatmul, random_tensors) {
    ov::element::Type net_type = std::get<1>(GetParam());
    size_t M = std::get<2>(GetParam());
    size_t N = std::get<3>(GetParam());
    size_t K = std::get<4>(GetParam());
    size_t lora_rank = std::get<5>(GetParam());

    targetStaticShapes = {{{{1, M, K}}, {{N, K}}}};
    run_test_random_tensors(net_type, lora_rank);
}

TEST_P(LoraPatternConvolution, random_tensors) {
    ov::element::Type net_type = std::get<1>(GetParam());
    size_t num_channels = std::get<2>(GetParam());
    size_t lora_rank = std::get<3>(GetParam());

    targetStaticShapes = {{{1, num_channels, 64, 64}}};
    run_test_random_tensors(net_type, lora_rank);
}

}  // namespace test
}  // namespace ov
