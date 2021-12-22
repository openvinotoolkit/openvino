// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/mul_conv_fusion.hpp"

namespace SubgraphTestsDefinitions {

TEST_P(MulConvFusion, CompareWithRefs) {
    Run();
}
} // namespace SubgraphTestsDefinitions
