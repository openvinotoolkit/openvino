// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/math_floor_f16.hpp"

namespace ov {
namespace test {

TEST_P(MathFloorF16Test, Inference) {
    run();
}

}  // namespace test
}  // namespace ov
