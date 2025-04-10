// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/inverse.hpp"

namespace {

using ov::test::InverseLayerTest;

const std::vector<std::vector<ov::test::InputShape>> input_shapes = {
    {{{5, 4, 4}, {{5, 4, 4}}}},
    {{{20, 3, 3}, {{20, 3, 3}}}},
    {{{ov::Dimension{1, 70}, -1}, {{3, 3}, {5, 5}, {4, 4}}}},
    {{{-1, ov::Dimension{1, 70}, -1}, {{3, 3, 3}, {4, 4, 4}, {5, 5, 5}}}}};

const auto shapes = testing::ValuesIn(input_shapes);

const auto dtypes = testing::Values(ov::element::f32, ov::element::f16, ov::element::bf16);

const auto adjoint = testing::Values(false, true);

const auto seed = testing::Values(1, 2, 3);

const auto device_cpu = testing::Values(ov::test::utils::DEVICE_CPU);

const auto params = ::testing::Combine(shapes, dtypes, adjoint, seed, device_cpu);

INSTANTIATE_TEST_SUITE_P(smoke_InverseStatic, InverseLayerTest, params, InverseLayerTest::getTestCaseName);
}  // namespace
