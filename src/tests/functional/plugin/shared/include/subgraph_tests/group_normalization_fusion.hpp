// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/group_normalization_fusion.hpp"

namespace ov {
namespace test {

TEST_P(GroupNormalizationFusionSubgraphTestsF, GroupNormalizationFusionSubgraphTests) {
    GroupNormalizationFusionSubgraphTestsF::run();
}

}  // namespace test
}  // namespace ov
