// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/strided_slice.hpp"

namespace SubgraphTestsDefinitions {

TEST_P(StridedSliceTest, CompareWithRefs){
    Run();
};
}  // namespace SubgraphTestsDefinitions
