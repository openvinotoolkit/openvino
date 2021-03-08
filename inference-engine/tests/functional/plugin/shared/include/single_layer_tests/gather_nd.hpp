// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_layer/gather_nd.hpp"

namespace LayerTestsDefinitions {

TEST_P(GatherNDLayerTest, CompareWithRefs) {
    Run();
}

}  // namespace LayerTestsDefinitions
