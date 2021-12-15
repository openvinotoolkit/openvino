// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/matmul_act_add.hpp"

namespace SubgraphTestsDefinitions {

TEST_P(MatMulActAddTest, CompareWithRefs) {
    Run();
};

} // namespace SubgraphTestsDefinitions
