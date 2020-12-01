// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "functional_test_utils/single_layer_test_classes/batch_norm.hpp"
#include "ngraph_functions/builders.hpp"

namespace LayerTestsDefinitions {

TEST_P(BatchNormLayerTest, CompareWithRefs) {
    Run();
}

}  // namespace LayerTestsDefinitions
