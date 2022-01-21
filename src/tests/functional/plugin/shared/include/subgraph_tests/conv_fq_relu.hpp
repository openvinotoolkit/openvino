// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/conv_fq_relu.hpp"

namespace SubgraphTestsDefinitions {

TEST_P(ConvFqReluTest, CompareWithRefs) {
    Run();
}

}  // namespace SubgraphTestsDefinitions
