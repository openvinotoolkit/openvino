// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/parameter_reshape_result.hpp"

namespace SubgraphTestsDefinitions {

TEST_P(ParamReshapeResult, CompareWithRefs) {
    Run();
};

}  // namespace SubgraphTestsDefinitions
