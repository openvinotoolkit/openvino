// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/const_conv_concat.hpp"

namespace SubgraphTestsDefinitions {

TEST_P(ConstConvConcatTest, CompareWithRefImpl) {
    LoadNetwork();
    GenerateInputs();
    Infer();
    // Create another copy of function for validation since some data will be changed by GNA plugin
    SetUp();
    Validate();
};
}  // namespace SubgraphTestsDefinitions