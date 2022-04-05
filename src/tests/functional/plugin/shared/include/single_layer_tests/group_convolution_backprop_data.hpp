// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_layer/group_convolution_backprop_data.hpp"

namespace LayerTestsDefinitions {

// DEPRECATED, remove this old API when KMB (#58495) and ARM (#58496) plugins are migrated to new API
TEST_P(GroupConvBackpropDataLayerTest, CompareWithRefs) {
    Run();
}

TEST_P(GroupConvBackpropLayerTest, CompareWithRefs) {
    Run();
}

}  // namespace LayerTestsDefinitions
