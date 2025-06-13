// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include "single_op/group_normalization.hpp"

namespace ov {
namespace test {

TEST_P(GroupNormalizationTest, CompareWithRefs) {
    run();
}

TEST_P(GroupNormalizationTest, CompareQueryModel) {
    query_model();
}

}  // namespace test
}  // namespace ov
