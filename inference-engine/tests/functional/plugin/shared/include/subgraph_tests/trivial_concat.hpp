// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/trivial_concat.hpp"

namespace SubgraphTestsDefinitions {

TEST_P(TrivialConcatLayerTest, CompareWithRefs) {
    Run();
};

}  // namespace SubgraphTestsDefinitions
