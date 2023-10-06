// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_layer/scatter_elements_update.hpp"

namespace LayerTestsDefinitions {

TEST_P(ScatterElementsUpdateLayerTest, CompareWithRefs) {
    Run();
};

TEST_P(ScatterElementsUpdate12LayerTest, CompareWithRefs) {
    Run();
};

}  // namespace LayerTestsDefinitions
