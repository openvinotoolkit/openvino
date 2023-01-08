// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/copy_before_squeeze.hpp"

namespace SubgraphTestsDefinitions {

TEST_P(CopyBeforeSqueezeTest, CompareWithRefs) {
    Run();
};

} // namespace SubgraphTestsDefinitions
