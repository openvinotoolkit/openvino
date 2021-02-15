// Copyright (C) 2020 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <shared_test_classes/single_layer/eltwise.hpp>

namespace LayerTestsDefinitions {
TEST_P(EltwiseLayerTest, EltwiseTests) {
    Run();
}
} // namespace LayerTestsDefinitions
