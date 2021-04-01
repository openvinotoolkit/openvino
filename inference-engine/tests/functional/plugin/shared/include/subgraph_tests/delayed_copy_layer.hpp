// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "shared_test_classes/subgraph/delayed_copy_layer.hpp"

namespace SubgraphTestsDefinitions {

TEST_P(DelayedCopyTest, CompareWithRefs) {
    Run();
};

} // namespace SubgraphTestsDefinitions
