// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_layer/fake_quantize.hpp"

namespace LayerTestsDefinitions {

TEST_P(FakeQuantizeLayerTest, CompareWithRefs) {
    Run();
    SKIP_IF_CURRENT_TEST_IS_DISABLED();

    if (BASE_SEED != USE_CLOCK_TIME &&
        BASE_SEED != USE_INCREMENTAL_SEED) {
        return;
    }

    size_t nIterations = 1;
    for (; nIterations != 0; nIterations--) {
        UpdateSeed();
        Infer();
        Validate();
    }
}

}  // namespace LayerTestsDefinitions
