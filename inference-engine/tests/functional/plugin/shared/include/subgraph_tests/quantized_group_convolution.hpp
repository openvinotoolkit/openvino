// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/quantized_group_convolution.hpp"

namespace SubgraphTestsDefinitions {

TEST_P(QuantGroupConvLayerTest, CompareWithRefs) {
    Run();
}

}  // namespace SubgraphTestsDefinitions