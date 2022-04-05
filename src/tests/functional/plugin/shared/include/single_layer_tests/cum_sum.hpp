// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_layer/cum_sum.hpp"

namespace LayerTestsDefinitions {

TEST_P(CumSumLayerTest, CompareWithRefs) {
    Run();
};

}  // namespace LayerTestsDefinitions
