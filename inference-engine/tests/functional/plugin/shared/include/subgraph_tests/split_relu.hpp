// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "shared_test_classes/subgraph/split_relu.hpp"

namespace LayerTestsDefinitions {

TEST_P(SplitRelu, CompareWithRefs){
    Run();
};

}  // namespace LayerTestsDefinitions
