// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/quantized_mat_mul.hpp"

namespace SubgraphTestsDefinitions {

TEST_P(QuantMatMulTest, CompareWithRefs) {
    Run();
};

}  // namespace SubgraphTestsDefinitions
