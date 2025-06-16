// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "functional_test_utils/skip_tests_config.hpp"
#include "shared_test_classes/subgraph/matmul_const_transposes_extraction.hpp"

namespace ov {
namespace test {

TEST_P(MatMulConstTransposesExtractionTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
}

TEST_P(QuantizedMatMulConstTransposesExtractionTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
}

}  // namespace test
}  // namespace ov
