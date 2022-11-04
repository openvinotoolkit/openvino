// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_layer/grid_sample.hpp"

namespace LayerTestsDefinitions {

TEST_P(GridSampleLayerTest, CompareWithRefs) {
    Run();
}

}  // namespace LayerTestsDefinitions
