// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/math_discontinuous_op_f16.hpp"

namespace ov::test {

TEST_P(MathDiscontinuousOpF16Test, Inference) {
    run();
}

}  // namespace ov::test
