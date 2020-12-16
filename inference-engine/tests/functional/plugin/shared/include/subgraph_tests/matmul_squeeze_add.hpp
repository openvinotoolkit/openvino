// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/matmul_squeeze_add.hpp"

namespace SubgraphTestsDefinitions {

TEST_P(MatmulSqueezeAddTest, CompareWithRefImpl) {
    Run();
};

}  // namespace SubgraphTestsDefinitions
