// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_op/gather_nd.hpp"

namespace ov {
namespace test {
TEST_P(GatherNDLayerTest, Inference) {
    run();
}

TEST_P(GatherND8LayerTest, Inference) {
    run();
}
}  // namespace test
}  // namespace ov
