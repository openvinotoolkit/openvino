// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_layer/random_uniform.hpp"

namespace LayerTestsDefinitions {

TEST_P(RandomUniformLayerTest, CompareWithRefs) {
    Run();
};

}  // namespace LayerTestsDefinitions
