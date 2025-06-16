// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_op/is_inf.hpp"

namespace ov {
namespace test {
TEST_P(IsInfLayerTest, Inference) {
    run();
}
}  // namespace test
}  // namespace ov
