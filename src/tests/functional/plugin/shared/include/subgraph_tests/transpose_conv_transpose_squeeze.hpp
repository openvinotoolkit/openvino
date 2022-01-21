// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/transpose_conv_transpose_squeeze.hpp"

namespace SubgraphTestsDefinitions {

TEST_P(TransposeConvTest, CompareWithRefImpl) {
    Run();
};

}  // namespace SubgraphTestsDefinitions
