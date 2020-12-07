// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "functional_test_utils/single_layer_test_classes/batch_to_space.hpp"

namespace LayerTestsDefinitions {

TEST_P(BatchToSpaceLayerTest, CompareWithRefs) {
    Run();
};

}  // namespace LayerTestsDefinitions
