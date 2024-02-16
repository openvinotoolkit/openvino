// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "single_op/reduce_ops.hpp"

namespace ov {
namespace test {
TEST_P(ReduceOpsLayerTest, Inference) {
    run();
}

TEST_P(ReduceOpsLayerWithSpecificInputTest, Inference) {
    run();
}
}  // namespace test
}  // namespace ov
