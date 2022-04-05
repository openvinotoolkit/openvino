// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/matmul_multiply_fusion.hpp"

namespace SubgraphTestsDefinitions {

TEST_P(MatMulMultiplyFusion, CompareWithRefs) {
    Run();
}

TEST_P(QuantizedMatMulMultiplyFusion, CompareWithRefs) {
    Run();
}
} // namespace SubgraphTestsDefinitions
