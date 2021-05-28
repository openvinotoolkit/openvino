// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/mvn_multiply_add.hpp"

namespace SubgraphTestsDefinitions {

TEST_P(MVNMultiplyAdd, CompareWithRefs){
    Run();
};

}  // namespace SubgraphTestsDefinitions
