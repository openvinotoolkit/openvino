// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_layer/deformable_psroi_pooling.hpp"

namespace LayerTestsDefinitions {

TEST_P(DeformablePSROIPoolingLayerTest, CompareWithRefs) {
    Run();
}

}  // namespace LayerTestsDefinitions
