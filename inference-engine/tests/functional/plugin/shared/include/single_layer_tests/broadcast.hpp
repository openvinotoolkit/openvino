// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "functional_test_utils/single_layer_test_classes/broadcast.hpp"

namespace LayerTestsDefinitions {

TEST_P(BroadcastLayerTest, CompareWithRefs) {
    Run();
}
}  // namespace LayerTestsDefinitions