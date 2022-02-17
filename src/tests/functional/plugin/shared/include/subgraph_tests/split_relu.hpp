// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/split_relu.hpp"

namespace SubgraphTestsDefinitions {

TEST_P(SplitRelu, CompareWithRefs){
    Run();
};

}  // namespace SubgraphTestsDefinitions
