// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "shared_test_classes/subgraph/scaleshift.hpp"

namespace LayerTestsDefinitions {

TEST_P(ScaleShiftLayerTest, CompareWithRefs){
    Run();
};
}  // namespace LayerTestsDefinitions
