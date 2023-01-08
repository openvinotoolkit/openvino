// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef TEST_FQ_WITH_MIXED_LEVELS_HPP
#define TEST_FQ_WITH_MIXED_LEVELS_HPP

#include "shared_test_classes/subgraph/fq_with_mixed_levels.hpp"

namespace SubgraphTestsDefinitions {

TEST_P(FqWithMixedLevelsTest, CompareWithRefImpl) {
    Run();
};

}  // namespace SubgraphTestsDefinitions

#endif // TEST_FQ_WITH_MIXED_LEVELS_HPP
