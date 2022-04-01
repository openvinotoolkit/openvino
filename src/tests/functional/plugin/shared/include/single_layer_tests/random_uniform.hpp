// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_layer/random_uniform.hpp"

namespace LayerTestsDefinitions {

TEST_P(RandomUniformLayerTest, CompareWithRefs) {
    Run();
};

TEST_P(RandomUniformLayerTest, QueryNetwork) {
    QueryNetwork();
}
}  // namespace LayerTestsDefinitions
