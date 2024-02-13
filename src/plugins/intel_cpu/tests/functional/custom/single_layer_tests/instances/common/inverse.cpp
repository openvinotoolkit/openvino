// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/inverse.hpp"

namespace {

using ov::test::InverseLayerTest;

const auto data = testing::Values(ov::Shape{10, 10}, ov::Shape{5, 7, 7}, ov::Shape{5, 4, 3, 3});

const auto dtypes = testing::Values(ElementType::f32, ElementType::f16, ElementType::bf16);

const auto adjoint = testing::Values(false, true);

const auto device_cpu = testing::Values(ov::test::utils::DEVICE_CPU);

const auto params_static = ::testing::Combine(shapes[0], dtypes, adjoint, device_cpu);
const auto params_dynamic = ::testing::Combine(shapes, dtypes, adjoint, device_cpu);

INSTANTIATE_TEST_SUITE_P(smoke_InverseStatic, InverseLayerTest, params_static, InverseLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_InverseDynamic, InverseLayerTest, params_dynamic, InverseLayerTest::getTestCaseName);
}  // namespace
