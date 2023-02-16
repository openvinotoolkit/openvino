// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/conv_fq_eltwise.hpp"

namespace SubgraphTestsDefinitions {

TEST_P(ConvFqEltwiseTest, CompareWithRefs) {
    Run();
}

}  // namespace SubgraphTestsDefinitions
