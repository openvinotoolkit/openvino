// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/reshape_permute_reshape.hpp"

namespace SubgraphTestsDefinitions {

TEST_P(ReshapePermuteReshape, CompareWithRefs) {
    Run();
}

} // namespace SubgraphTestsDefinitions
