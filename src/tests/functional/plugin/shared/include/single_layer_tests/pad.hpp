// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_layer/pad.hpp"

namespace LayerTestsDefinitions {

TEST_P(PadLayerTest, CompareWithRefs) {
    Run();
}

TEST_P(PadLayerTest12, CompareWithRefs) {
    Run();
}

}  // namespace LayerTestsDefinitions
