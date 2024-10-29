// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/lora_pattern.hpp"

namespace ov {
namespace test {

TEST_P(LoraPatternMatmul, empty_tensors) {
    targetStaticShapes = {{{{1, 20, K}}, {{N, K}}}};
    run_test_empty_tensors();
}

TEST_P(LoraPatternConvolution, empty_tensors) {
    targetStaticShapes = {{{1, num_channels, 64, 64}}};
    run_test_empty_tensors();
}

TEST_P(LoraPatternMatmul, random_tensors) {
    targetStaticShapes = {{{{1, 20, K}}, {{N, K}}}};
    run_test_random_tensors();
}

TEST_P(LoraPatternConvolution, random_tensors) {
    targetStaticShapes = {{{1, num_channels, 64, 64}}};
    run_test_random_tensors();
}

}  // namespace test
}  // namespace ov