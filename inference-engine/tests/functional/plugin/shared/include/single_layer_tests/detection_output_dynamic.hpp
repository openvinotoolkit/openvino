// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_layer/detection_output_dynamic.hpp"

namespace LayerTestsDefinitions {

TEST_P(DetectionOutputDynamicLayerTest, CompareWithRefs) {
    run();
};

}  // namespace LayerTestsDefinitions
