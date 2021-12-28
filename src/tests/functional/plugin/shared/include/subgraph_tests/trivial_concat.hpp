// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/trivial_concat.hpp"

namespace SubgraphTestsDefinitions {

TEST_P(TrivialConcatLayerTest, CompareWithRefs) {
    Run();
};

TEST_P(TrivialConcatLayerTest2Inputs, CompareWithRefs) {
    Run();
};

TEST_P(TrivialConcatLayerTest_MultipleInputs, CompareWithRefs) {
    Run();
};

}  // namespace SubgraphTestsDefinitions
