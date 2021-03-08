// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "shared_test_classes/single_layer/batch_to_space.hpp"

namespace LayerTestsDefinitions {

TEST_P(BatchToSpaceLayerTest, CompareWithRefs) {
    Run();
};

}  // namespace LayerTestsDefinitions
