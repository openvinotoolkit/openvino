// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_layer/gather.hpp"

namespace LayerTestsDefinitions {

TEST_P(GatherLayerTest, CompareWithRefs) {
    Run();
};

TEST_P(Gather7LayerTest, CompareWithRefs) {
    Run();
};

TEST_P(Gather8LayerTest, CompareWithRefs) {
    Run();
};

TEST_P(Gather8IndiceScalarLayerTest, CompareWithRefs) {
    Run();
};

TEST_P(Gather8withIndicesDataLayerTest, CompareWithRefs) {
    Run();
};

}  // namespace LayerTestsDefinitions
