// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "functional_test_utils/single_layer_test_classes/concat.hpp"

namespace LayerTestsDefinitions {

TEST_P(ConcatLayerTest, CompareWithRefs) {
    Run();
};

}  // namespace LayerTestsDefinitions
