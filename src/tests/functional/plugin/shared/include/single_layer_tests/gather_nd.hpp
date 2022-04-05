// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_layer/gather_nd.hpp"

namespace LayerTestsDefinitions {

TEST_P(GatherNDLayerTest, CompareWithRefs) {
    Run();
}

TEST_P(GatherND8LayerTest, CompareWithRefs) {
    Run();
}

}  // namespace LayerTestsDefinitions
