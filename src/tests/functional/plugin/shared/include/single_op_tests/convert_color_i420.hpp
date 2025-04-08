// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_op/convert_color_i420.hpp"

namespace ov {
namespace test {
TEST_P(ConvertColorI420LayerTest, Inference) {
    run();
}
} // namespace test
} // namespace ov
