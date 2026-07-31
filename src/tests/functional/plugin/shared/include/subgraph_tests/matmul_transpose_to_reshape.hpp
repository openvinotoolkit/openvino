// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/matmul_transpose_to_reshape.hpp"

namespace ov {
namespace test {

TEST_P(MatMulTransposeToReshape, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();

    const auto runtime_model = compiledModel.get_runtime_model();
    CheckNumberOfNodesWithTypes(runtime_model, {"FullyConnected"}, 1);
    CheckNumberOfNodesWithTypes(runtime_model, {"Transpose"}, 0);
    CheckNumberOfNodesWithTypes(runtime_model, {"Permute"}, 0);
}

}  // namespace test
}  // namespace ov
