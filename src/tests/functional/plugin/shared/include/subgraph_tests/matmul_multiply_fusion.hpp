// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "functional_test_utils/skip_tests_config.hpp"
#include "shared_test_classes/subgraph/matmul_multiply_fusion.hpp"

namespace ov {
namespace test {

TEST_P(MatMulMultiplyFusion, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
}

TEST_P(QuantizedMatMulMultiplyFusion, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
}

}  // namespace test
}  // namespace ov
