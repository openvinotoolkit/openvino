// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_layer/space_to_depth.hpp"

namespace LayerTestsDefinitions {

TEST_P(SpaceToDepthLayerTest, CompareWithRefs) {
    Run();
};

}  // namespace LayerTestsDefinitions