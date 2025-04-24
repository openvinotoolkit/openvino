// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_op/topk.hpp"

namespace ov {
namespace test {
TEST_P(TopKLayerTest, Inference) {
    run();
}

TEST_P(TopK11LayerTest, Inference) {
    run();
}
}  // namespace test
}  // namespace ov
