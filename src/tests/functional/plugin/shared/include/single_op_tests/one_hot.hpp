// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_op/one_hot.hpp"

namespace ov {
namespace test {
TEST_P(OneHot1LayerTest, Inference) {
    run();
}

TEST_P(OneHot16LayerTest, Inference) {
    run();
}
}  // namespace test
}  // namespace ov
