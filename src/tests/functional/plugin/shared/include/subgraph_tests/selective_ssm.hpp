// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/selective_ssm.hpp"

namespace ov::test {

TEST_P(SelectiveSSM, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
    auto runtime_model = compiledModel.get_runtime_model();
    CheckNumberOfNodesWithType(runtime_model, {"SelectiveSSM"}, 1);
    CheckNumberOfNodesWithType(runtime_model, {"Loop"}, 0);
}

}  // namespace ov::test
