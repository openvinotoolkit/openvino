// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_layer/non_max_suppression.hpp"

namespace LayerTestsDefinitions {

TEST_P(NmsLayerTest, CompareWithRefs) {
    Run();
};

}  // namespace LayerTestsDefinitions
