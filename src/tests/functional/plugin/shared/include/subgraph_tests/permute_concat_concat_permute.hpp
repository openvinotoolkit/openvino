// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/permute_concat_concat_permute.hpp"

namespace SubgraphTestsDefinitions {

TEST_P(PermuteConcatConcatPermute, CompareWithRefs) {
    Run();
}

}  // namespace SubgraphTestsDefinitions
