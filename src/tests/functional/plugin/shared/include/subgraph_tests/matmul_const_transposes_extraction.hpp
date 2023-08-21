// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/matmul_const_transposes_extraction.hpp"

namespace SubgraphTestsDefinitions {

TEST_P(MatMulConstTransposesExtractionTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    Run();
}

TEST_P(QuantizedMatMulConstTransposesExtractionTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    Run();
}

} // namespace SubgraphTestsDefinitions
