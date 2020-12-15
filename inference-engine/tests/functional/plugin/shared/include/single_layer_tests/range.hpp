// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_layer/range.hpp"

namespace LayerTestsDefinitions {

TEST_P(RangeNumpyLayerTest, CompareWithRefs) {
    Run();
}

}  // namespace LayerTestsDefinitions