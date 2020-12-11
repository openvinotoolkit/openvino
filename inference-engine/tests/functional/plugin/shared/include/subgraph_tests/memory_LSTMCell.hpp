// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "shared_test_classes/subgraph/memory_LSTMCell.hpp"

namespace SubgraphTestsDefinitions {

TEST_P(MemoryLSTMCellTest, CompareWithRefs) {
    Run();
};

TEST_P(MemoryLSTMCellTest, CompareWithRefs_LowLatencyTransformation) {
    RunLowLatency();
};

TEST_P(MemoryLSTMCellTest, CompareWithRefs_LowLatencyRegularAPITransformation) {
    RunLowLatency(true);
};

} // namespace SubgraphTestsDefinitions
