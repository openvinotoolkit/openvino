// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/split_conv_concat.hpp"

namespace SubgraphTestsDefinitions {

TEST_P(SplitConvConcat, CompareWithRefImpl) {
    Run();
};

}  // namespace SubgraphTestsDefinitions