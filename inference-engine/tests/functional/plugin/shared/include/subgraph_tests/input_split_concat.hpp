// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/input_split_concat.hpp"

namespace SubgraphTestsDefinitions {

TEST_P(InputSplitConcatTest, CompareWithRefImpl) {
    Run();
};

}  // namespace SubgraphTestsDefinitions
