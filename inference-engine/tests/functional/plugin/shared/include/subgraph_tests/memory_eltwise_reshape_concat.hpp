// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "shared_test_classes/subgraph/memory_eltwise_reshape_concat.hpp"

namespace SubgraphTestsDefinitions {

TEST_P(MemoryEltwiseReshapeConcatTest, CompareWithRefs) {
    Run();
};

} // namespace SubgraphTestsDefinitions
