// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_layer/scatter_ND_update.hpp"

namespace LayerTestsDefinitions {

TEST_P(ScatterNDUpdateLayerTest, CompareWithRefs) {
    Run();
};

}  // namespace LayerTestsDefinitions