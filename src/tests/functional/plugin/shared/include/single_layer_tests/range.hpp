// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_layer/range.hpp"

namespace LayerTestsDefinitions {

TEST_P(RangeNumpyLayerTest, CompareWithRefs) {
    Run();
}

TEST_P(RangeLayerTest, CompareWithRefs) {
    Run();
}

TEST_P(RangeNumpyLayerTest, QueryNetwork) {
    QueryNetwork();
}

TEST_P(RangeLayerTest, QueryNetwork) {
    QueryNetwork();
}
}  // namespace LayerTestsDefinitions