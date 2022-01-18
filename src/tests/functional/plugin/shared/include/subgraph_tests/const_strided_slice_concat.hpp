// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/const_strided_slice_concat.hpp"

namespace SubgraphTestsDefinitions {

TEST_P(ConstStridedSliceConcatTest, CompareWithRefImpl) {
    Run();
};
}  // namespace SubgraphTestsDefinitions
