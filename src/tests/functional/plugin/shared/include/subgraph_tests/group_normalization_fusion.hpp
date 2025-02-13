// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/group_normalization_fusion.hpp"

namespace ov {
namespace test {

TEST_P(GroupNormalizationFusionSubgraphTestsF_f32, GroupNormalizationFusionSubgraphTests_f32) {
    GroupNormalizationFusionSubgraphTestsF_f32::run();
}

TEST_P(GroupNormalizationFusionSubgraphTestsF_f16, GroupNormalizationFusionSubgraphTests_f16) {
    GroupNormalizationFusionSubgraphTestsF_f16::run();
}

TEST_P(GroupNormalizationFusionSubgraphTestsF_bf16, GroupNormalizationFusionSubgraphTests_bf16) {
    GroupNormalizationFusionSubgraphTestsF_bf16::run();
}

TEST_P(GroupNormalizationFusionSubgraphTestsF_u8, GroupNormalizationFusionSubgraphTests_u8) {
    GroupNormalizationFusionSubgraphTestsF_u8::run();
}

TEST_P(GroupNormalizationFusionSubgraphTestsF_u16, GroupNormalizationFusionSubgraphTests_u16) {
    GroupNormalizationFusionSubgraphTestsF_u16::run();
}

TEST_P(GroupNormalizationFusionSubgraphTestsF_u32, GroupNormalizationFusionSubgraphTests_u32) {
    GroupNormalizationFusionSubgraphTestsF_u32::run();
}

TEST_P(GroupNormalizationFusionSubgraphTestsF_u64, GroupNormalizationFusionSubgraphTests_u64) {
    GroupNormalizationFusionSubgraphTestsF_u64::run();
}

TEST_P(GroupNormalizationFusionSubgraphTestsF_i8, GroupNormalizationFusionSubgraphTests_i8) {
    GroupNormalizationFusionSubgraphTestsF_i8::run();
}

TEST_P(GroupNormalizationFusionSubgraphTestsF_i16, GroupNormalizationFusionSubgraphTests_i16) {
    GroupNormalizationFusionSubgraphTestsF_i16::run();
}

TEST_P(GroupNormalizationFusionSubgraphTestsF_i32, GroupNormalizationFusionSubgraphTests_i32) {
    GroupNormalizationFusionSubgraphTestsF_i32::run();
}

TEST_P(GroupNormalizationFusionSubgraphTestsF_i64, GroupNormalizationFusionSubgraphTests_i64) {
    GroupNormalizationFusionSubgraphTestsF_i64::run();
}

TEST_P(GroupNormalizationFusionSubgraphTestsF_f8e4m3, GroupNormalizationFusionSubgraphTests_f8e4m3) {
    GroupNormalizationFusionSubgraphTestsF_f8e4m3::run();
}

TEST_P(GroupNormalizationFusionSubgraphTestsF_f8e5m2, GroupNormalizationFusionSubgraphTests_f8e5m2) {
    GroupNormalizationFusionSubgraphTestsF_f8e5m2::run();
}

TEST_P(GroupNormalizationFusionSubgraphTestsF_f4e2m1, GroupNormalizationFusionSubgraphTests_f4e2m1) {
    GroupNormalizationFusionSubgraphTestsF_f4e2m1::run();
}

TEST_P(GroupNormalizationFusionSubgraphTestsF_f8e8m0, GroupNormalizationFusionSubgraphTests_f8e8m0) {
    GroupNormalizationFusionSubgraphTestsF_f8e8m0::run();
}

}  // namespace test
}  // namespace ov
