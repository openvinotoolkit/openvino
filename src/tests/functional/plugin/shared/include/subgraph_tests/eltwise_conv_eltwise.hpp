// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/eltwise_conv_eltwise.hpp"

namespace SubgraphTestsDefinitions {

TEST_P(EltwiseAfterConvTest, CompareWithRefImpl) {
    LoadNetwork();
    GenerateInputs();
    Infer();
    // Create another copy of function for validation since some data will be changed by GNA plugin
    SetUp();
    Validate();
};

TEST_P(EltwiseBeforeConvTest, CompareWithRefImpl) {
    LoadNetwork();
    GenerateInputs();
    Infer();
    // Create another copy of function for validation since some data will be changed by GNA plugin
    SetUp();
    Validate();
};

TEST_P(EltwiseWithTwoConvsAsInputsTest, CompareWithRefImpl) {
    Run();
};

}  // namespace SubgraphTestsDefinitions