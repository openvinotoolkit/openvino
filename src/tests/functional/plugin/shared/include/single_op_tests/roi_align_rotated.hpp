// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_op/roi_align_rotated.hpp"

namespace ov {
namespace test {
TEST_P(ROIAlignRotatedLayerTest, Inference) {
    run();
}
}  // namespace test
}  // namespace ov
