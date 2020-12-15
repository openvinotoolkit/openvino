// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "shared_test_classes/subgraph/multiple_concat.hpp"

namespace SubgraphTestsDefinitions {

TEST_P(MultipleConcatTest, CompareWithRefs) {
    Run();
};

} // namespace SubgraphTestsDefinitions
