// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_layer/custom_operation.hpp"
#include "ngraph_functions/builders.hpp"

namespace LayerTestsDefinitions {

TEST_P(CustomOpLayerTest, CompareWithRefs) {
    Run();
}

}  // namespace LayerTestsDefinitions

