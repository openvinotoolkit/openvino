// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_layer/interpolate.hpp"

namespace LayerTestsDefinitions {

TEST_P(InterpolateLayerTest, CompareWithRefs) {
    Run();
}

using Interpolate11LayerTest = v11::InterpolateLayerTest;

TEST_P(Interpolate11LayerTest, CompareWithRefs) {
    Run();
}

}  // namespace LayerTestsDefinitions
