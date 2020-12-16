// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_layer/strided_slice.hpp"

namespace LayerTestsDefinitions {
TEST_P(StridedSliceLayerTest, CompareWithRefs) {
    Run();
}
}  // namespace LayerTestsDefinitions
