// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_layer/gather.hpp"

namespace LayerTestsDefinitions {

TEST_P(GatherLayerTest, CompareWithRefs) {
    Run();
};

}  // namespace LayerTestsDefinitions
