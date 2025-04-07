// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_op/non_max_suppression.hpp"

namespace ov {
namespace test {
TEST_P(NmsLayerTest, Inference) {
    run();
};

TEST_P(Nms9LayerTest, Inference) {
    run();
};
}  // namespace test
}  // namespace ov
