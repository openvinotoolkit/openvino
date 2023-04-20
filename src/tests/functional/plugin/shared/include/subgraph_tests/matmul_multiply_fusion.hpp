// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/matmul_multiply_fusion.hpp"

namespace SubgraphTestsDefinitions {

TEST_P(MatMulMultiplyFusion, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    Run();
}

TEST_P(QuantizedMatMulMultiplyFusion, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    Run();
}
} // namespace SubgraphTestsDefinitions
