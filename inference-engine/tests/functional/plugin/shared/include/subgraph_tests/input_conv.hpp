// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/input_conv.hpp"

namespace SubgraphTestsDefinitions {

TEST_P(InputConvTest, CompareWithRefImpl) {
    Run();
};

}  // namespace SubgraphTestsDefinitions
