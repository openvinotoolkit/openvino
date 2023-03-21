// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/permute_concat_concat_permute.hpp"

namespace SubgraphTestsDefinitions {

using PermuteConcatConcatPermuteNeg = PermuteConcatConcatPermute;

TEST_P(PermuteConcatConcatPermute, CompareWithRefs) {
    Run();
}

TEST_P(PermuteConcatConcatPermuteNeg, CompareWithRefs) {
    ExpectLoadNetworkToThrow("type: Concat, and concatenation axis(");
}

}  // namespace SubgraphTestsDefinitions
