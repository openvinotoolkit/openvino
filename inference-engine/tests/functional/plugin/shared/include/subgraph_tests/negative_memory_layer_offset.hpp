// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "shared_test_classes/subgraph/negative_memory_layer_offset.hpp"

namespace SubgraphTestsDefinitions {

TEST_P(NegativeMemoryOffsetTest, CompareWithRefs) {
    Run();
};

} // namespace SubgraphTestsDefinitions
