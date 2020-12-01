// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "functional_test_utils/single_layer_test_classes/variadic_split.hpp"

namespace LayerTestsDefinitions {

TEST_P(VariadicSplitLayerTest, CompareWithRefs) {
    Run();
}

}  // namespace LayerTestsDefinitions