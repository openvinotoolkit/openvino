// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/split_to_concat_with_3_inputs.hpp"

namespace SubgraphTestsDefinitions {

TEST_P(SplitConcatWith3InputsTest, CompareWithRefImpl) {
    Run();
};

}  // namespace SubgraphTestsDefinitions