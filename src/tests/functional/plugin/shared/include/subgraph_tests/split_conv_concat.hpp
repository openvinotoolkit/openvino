// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/split_conv_concat.hpp"

namespace SubgraphTestsDefinitions {

TEST_P(SplitConvConcat, CompareWithRefImpl) {
    Run();
};

TEST_P(SplitConvConcat, QueryNetwork) {
    QueryNetwork();
}

}  // namespace SubgraphTestsDefinitions
