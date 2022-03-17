// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/fc_conv_fc.hpp"

namespace SubgraphTestsDefinitions {

TEST_P(FcAfterConvTest, CompareWithRefImpl) {
    Run();
};

TEST_P(FcBeforeConvTest, CompareWithRefImpl) {
    Run();
};

TEST_P(FcBetweenConvsTest, CompareWithRefImpl) {
    Run();
};

}  // namespace SubgraphTestsDefinitions