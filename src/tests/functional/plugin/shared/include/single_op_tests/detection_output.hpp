// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "single_op/detection_output.hpp"

namespace ov {
namespace test {
TEST_P(DetectionOutputLayerTest, Inference) {
    run();
};
}  // namespace test
}  // namespace ov
