// Copyright (C) 2020 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_layer/roi_align.hpp"

namespace LayerTestsDefinitions {

TEST_P(ROIAlignLayerTest, CompareWithRefs) {
    Run();
}

}  // namespace LayerTestsDefinitions
