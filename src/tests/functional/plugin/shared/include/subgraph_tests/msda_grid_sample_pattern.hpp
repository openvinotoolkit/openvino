// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "shared_test_classes/subgraph/msda_grid_sample_pattern.hpp"

namespace ov {
namespace test {

TEST_P(MSDAGridSamplePattern, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
    CheckNumberOfNodesWithType(compiledModel, "msda", 1);
}

}  // namespace test
}  // namespace ov
