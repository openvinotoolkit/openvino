// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/quantized_convolution_backprop_data.hpp"

namespace SubgraphTestsDefinitions {

TEST_P(QuantConvBackpropDataLayerTest, CompareWithRefs) {
    Run();
}

}  // namespace SubgraphTestsDefinitions