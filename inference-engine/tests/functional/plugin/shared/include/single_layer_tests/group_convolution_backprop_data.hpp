// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_layer/group_convolution_backprop_data.hpp"

namespace LayerTestsDefinitions {

TEST_P(GroupConvBackpropDataLayerTest, CompareWithRefs) {
    Run();
}

}  // namespace LayerTestsDefinitions