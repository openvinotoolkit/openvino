// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_layer/mvn.hpp"

namespace LayerTestsDefinitions {

// DEPRECATED, remove MvnLayerTest when KMB and ARM plugin will switch to use Mvn1LayerTest (#60420)
TEST_P(MvnLayerTest, CompareWithRefs) {
    Run();
};

TEST_P(Mvn1LayerTest, CompareWithRefs) {
    Run();
};

TEST_P(Mvn6LayerTest, CompareWithRefs) {
    Run();
};

}  // namespace LayerTestsDefinitions