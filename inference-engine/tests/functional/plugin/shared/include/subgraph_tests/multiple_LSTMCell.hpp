// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "shared_test_classes/subgraph/multiple_LSTMCell.hpp"

namespace SubgraphTestsDefinitions {

TEST_P(MultipleLSTMCellTest, CompareWithRefs) {
    Run();
};

TEST_P(MultipleLSTMCellTest, CompareWithRefs_LowLatencyTransformation) {
    RunLowLatency();
};

TEST_P(MultipleLSTMCellTest, CompareWithRefs_LowLatencyRegularAPITransformation) {
    RunLowLatency(true);
};

} // namespace SubgraphTestsDefinitions
