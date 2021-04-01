// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "shared_test_classes/subgraph/conv_eltwise_fusion.hpp"

namespace SubgraphTestsDefinitions {

TEST_P(ConvEltwiseFusion, CompareWithRefs) {
    Run();
}
} // namespace SubgraphTestsDefinitions
