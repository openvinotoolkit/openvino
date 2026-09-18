// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_op/grouped_matmul.hpp"

namespace ov {
namespace test {
TEST_P(GroupedMatMulLayerTest, Inference) {
    run();
}
TEST_P(GroupedMatMulCompressedLayerTest, Inference) {
    run();
}
}  // namespace test
}  // namespace ov
