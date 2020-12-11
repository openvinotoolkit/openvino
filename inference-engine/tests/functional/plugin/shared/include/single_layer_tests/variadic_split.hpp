// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_layer/variadic_split.hpp"

namespace LayerTestsDefinitions {

TEST_P(VariadicSplitLayerTest, CompareWithRefs) {
    Run();
}

}  // namespace LayerTestsDefinitions
