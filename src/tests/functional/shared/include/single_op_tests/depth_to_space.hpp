// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "single_op/depth_to_space.hpp"

namespace ov {
namespace test {
TEST_P(DepthToSpaceLayerTest, Inference) {
    run();
};
}  // namespace test
}  // namespace ov
