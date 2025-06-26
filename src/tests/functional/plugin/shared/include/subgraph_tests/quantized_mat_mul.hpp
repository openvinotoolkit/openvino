// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "subgraph/quantized_mat_mul.hpp"

namespace ov {
namespace test {

TEST_P(QuantMatMulTest, CompareWithRefs) {
    run();
};

}  // namespace test
}  // namespace ov
