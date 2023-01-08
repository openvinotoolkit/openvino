// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// DEPRECATED, can't be removed currently due to arm and kmb-plugin dependency (#55568)
#pragma once

#include "shared_test_classes/single_layer/convolution_backprop_data.hpp"

namespace LayerTestsDefinitions {

TEST_P(ConvolutionBackpropDataLayerTest, CompareWithRefs) {
    Run();
}

}
