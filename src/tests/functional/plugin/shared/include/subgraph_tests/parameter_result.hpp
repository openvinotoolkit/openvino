// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/parameter_result.hpp"

namespace SubgraphTestsDefinitions {

TEST_P(ParameterResultSubgraphTestLegacyApi, CompareWithRefs) {
    Run();
}

}  // namespace SubgraphTestsDefinitions

namespace ov {
namespace test {

TEST_P(ParameterResultSubgraphTest, CompareWithRefs) {
    run();
}

}  // namespace test
}  // namespace ov
