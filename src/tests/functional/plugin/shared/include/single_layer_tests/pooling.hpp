// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_layer/pooling.hpp"

namespace LayerTestsDefinitions {

TEST_P(PoolingLayerTest, CompareWithRefs) {
    Run();
}

TEST_P(GlobalPoolingLayerTest, CompareWithRefs) {
    Run();

    if (targetDevice == std::string{ov::test::utils::DEVICE_GPU}) {
        PluginCache::get().reset();
    }
}

TEST_P(MaxPoolingV8LayerTest, CompareWithRefs) {
    Run();
}

}  // namespace LayerTestsDefinitions
