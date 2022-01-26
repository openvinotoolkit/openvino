// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_layer/activation.hpp"

namespace LayerTestsDefinitions {

TEST_P(ActivationLayerTest, CompareWithRefs) {
    Run();
}

TEST_P(ActivationParamLayerTest, CompareWithRefs) {
    Run();
}

TEST_P(ActivationDynamicLayerTest, CompareWithRefs) {
    Run();
}

TEST_P(ActivationLayerTest, QueryNetwork) {
    QueryNetwork();
}

TEST_P(ActivationParamLayerTest, QueryNetwork) {
    QueryNetwork();
}

TEST_P(ActivationDynamicLayerTest, QueryNetwork) {
    QueryNetwork();
}

}  // namespace LayerTestsDefinitions
