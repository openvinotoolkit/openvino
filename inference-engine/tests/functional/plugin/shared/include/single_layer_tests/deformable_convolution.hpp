// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_layer/deformable_convolution.hpp"

namespace LayerTestsDefinitions {

TEST_P(DeformableConvolutionLayerTest, CompareWithRefs) {
    Run();
}

}  // namespace LayerTestsDefinitions
