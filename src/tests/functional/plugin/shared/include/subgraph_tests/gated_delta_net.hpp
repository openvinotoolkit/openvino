// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/gated_delta_net.hpp"

namespace ov {
namespace test {

TEST_P(GatedDeltaNet, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
    auto function = compiledModel.get_runtime_model();
};
}  // namespace test
}  // namespace ov