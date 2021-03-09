// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/split_conv.hpp"

namespace SubgraphTestsDefinitions {

TEST_P(SplitConvTest, CompareWithRefImpl) {
    Run();
};

}  // namespace SubgraphTestsDefinitions