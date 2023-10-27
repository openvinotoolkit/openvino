// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include "shared_test_classes/single_layer/group_normalization.hpp"

namespace ov {
namespace test {
namespace subgraph {

TEST_P(GroupNormalizationTest, CompareWithRefs) {
    run();
}

TEST_P(GroupNormalizationTest, CompareQueryModel) {
    query_model();
}

}  // namespace subgraph
}  // namespace test
}  // namespace ov
