// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "shared_test_classes/single_layer/result.hpp"

namespace LayerTestsDefinitions {

TEST_P(ResultLayerTest, CompareWithRefs) {
    Run();
}
}  // namespace LayerTestsDefinitions
