// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/inverse.hpp"

namespace {

using ov::test::InverseLayerTest;

const std::vector<std::vector<ov::test::InputShape>> input_shapes = {
    {{{ov::Dimension{1, 70}, -1}, {{10, 10}, {7, 7}, {4, 4}}}},
    {{{-1, ov::Dimension{1, 70}, -1}, {{2, 10, 10}, {10, 7, 7}, {20, 4, 4}}}}};
const auto shapes = testing::ValuesIn(input_shapes);

const auto dtypes = testing::Values(ov::element::f32, ov::element::f16, ov::element::bf16);

const auto adjoint = testing::Values(false, true);

const auto test_static = testing::Values(true);
const auto test_dynamic = testing::Values(false);

const auto seed = testing::Values(1, 2, 3);

const auto params_dynamic = ::testing::Combine(shapes, dtypes, adjoint, test_dynamic, seed, device_cpu);

INSTANTIATE_TEST_SUITE_P(smoke_InverseStatic, InverseLayerTest, params_static, InverseLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_InverseDynamic, InverseLayerTest, params_dynamic, InverseLayerTest::getTestCaseName);
}  // namespace
