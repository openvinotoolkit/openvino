// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/matmul_const_transposes_extraction.hpp"

namespace SubgraphTestsDefinitions {

TEST_P(MatMulConstTransposesExtractionTest, CompareWithRefs) {
    Run();
}

TEST_P(QuantizedMatMulConstTransposesExtractionTest, CompareWithRefs) {
    Run();
}

} // namespace SubgraphTestsDefinitions
