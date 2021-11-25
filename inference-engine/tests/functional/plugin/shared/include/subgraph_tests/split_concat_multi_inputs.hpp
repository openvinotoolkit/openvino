// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/split_concat_multi_inputs.hpp"

namespace SubgraphTestsDefinitions {

TEST_P(SplitConcatMultiInputsTest, CompareWithRefs) {
    Run();
};
}  // namespace SubgraphTestsDefinitions