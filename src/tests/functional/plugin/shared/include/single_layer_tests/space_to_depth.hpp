// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_layer/space_to_depth.hpp"

namespace LayerTestsDefinitions {

TEST_P(SpaceToDepthLayerTest, CompareWithRefs) {
    Run();
};

TEST_P(SpaceToDepthLayerTest, QueryNetwork) {
    QueryNetwork();
}
}  // namespace LayerTestsDefinitions