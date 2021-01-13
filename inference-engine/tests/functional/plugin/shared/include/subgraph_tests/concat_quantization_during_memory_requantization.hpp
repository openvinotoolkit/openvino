// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "shared_test_classes/subgraph/concat_quantization_during_memory_requantization.hpp"

namespace SubgraphTestsDefinitions {

TEST_P(ConcatQuantDuringMemoryRequantTest, CompareWithRefs) {
    Run();
};

} // namespace SubgraphTestsDefinitions
