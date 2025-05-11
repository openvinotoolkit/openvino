// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_op/ctc_greedy_decoder.hpp"

namespace ov {
namespace test {
TEST_P(CTCGreedyDecoderLayerTest, Inference) {
    run();
};
}  // namespace test
}  // namespace ov
