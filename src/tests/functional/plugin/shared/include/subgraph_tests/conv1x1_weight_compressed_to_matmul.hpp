// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/conv1x1_weight_compressed_to_matmul.hpp"

namespace ov {
namespace test {

TEST_P(Conv1x1WeightCompressedToMatmulTest, Inference) {
    run();
}

}  // namespace test
}  // namespace ov
