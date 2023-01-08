// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/parameter_result.hpp"

namespace SubgraphTestsDefinitions {

TEST_P(ParameterResultSubgraphTestLegacyApi, CompareWithRefs) {
    Run();
}

TEST_P(ParameterResultSubgraphTest, CompareWithRefs) {
    run();
}

}  // namespace SubgraphTestsDefinitions
