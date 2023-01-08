// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/reshape_permute_conv_permute_reshape_act.hpp"

namespace SubgraphTestsDefinitions {

TEST_P(ConvReshapeAct, CompareWithRefs) {
    Run();
}

}  // namespace SubgraphTestsDefinitions
