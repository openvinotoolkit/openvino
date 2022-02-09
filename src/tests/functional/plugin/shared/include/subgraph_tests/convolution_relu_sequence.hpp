// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/convolution_relu_sequence.hpp"

namespace SubgraphTestsDefinitions {

TEST_P(ConvolutionReluSequenceTest, CompareWithRefImpl) {
    Run();
};

}  // namespace SubgraphTestsDefinitions
