// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/gated_delta_net.hpp"

namespace ov::test {

TEST_P(GatedDeltaNet, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
    auto function = compiledModel.get_runtime_model();
    CheckNumberOfNodesWithType(function, {"GatedDeltaNet"}, 1);
    CheckNumberOfNodesWithType(function, {"Transpose"}, 0);
    CheckNumberOfNodesWithType(function, {"Concat"}, 0);
    CheckNumberOfNodesWithType(function, {"ReduceSum"}, 0);
    CheckNumberOfNodesWithType(function, {"Multiply"}, 0);
    CheckNumberOfNodesWithType(function, {"Divide"}, 0);
};
}  // namespace ov::test
