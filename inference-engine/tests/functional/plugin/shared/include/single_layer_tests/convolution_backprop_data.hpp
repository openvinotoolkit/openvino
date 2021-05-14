// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// DEPRECATED, this file shall be removed when kmb_plugin will switch to use new API from "convolution_backprop.hpp" (#55568)

#pragma once

#include "shared_test_classes/single_layer/convolution_backprop_data.hpp"

namespace LayerTestsDefinitions {

TEST_P(ConvolutionBackpropDataLayerTest, CompareWithRefs) {
    Run();
}

}
