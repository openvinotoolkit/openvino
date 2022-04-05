// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/const_strided_slice_concat.hpp"

namespace SubgraphTestsDefinitions {

TEST_P(ConstStridedSliceConcatTest, CompareWithRefImpl) {
    Run();
};
}  // namespace SubgraphTestsDefinitions
