// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/parameter_shapeof_result.hpp"

namespace SubgraphTestsDefinitions {

TEST_P(ParameterShapeOfResultSubgraphTest, CompareWithRefs) {
    Run();
}

}  // namespace SubgraphTestsDefinitions
