// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_op/grid_sample.hpp"

namespace ov {
namespace test {
TEST_P(GridSampleLayerTest, Inference) {
    run();
}
}  // namespace test
}  // namespace ov
