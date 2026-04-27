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
    CheckNumberOfNodesWithTypes(function, {"GatedDeltaNet"}, 1);
    CheckNumberOfNodesWithTypes(function, {"Transpose"}, 0);
    CheckNumberOfNodesWithTypes(function, {"Concat"}, 0);
    CheckNumberOfNodesWithTypes(function, {"ReduceSum"}, 0);
    CheckNumberOfNodesWithTypes(function, {"Multiply"}, 0);
    CheckNumberOfNodesWithTypes(function, {"Divide"}, 0);
};
}  // namespace ov::test
