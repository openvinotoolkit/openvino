// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/inverse.hpp"

#include <openvino/core/type/element_type.hpp>
#include <openvino/runtime/tensor.hpp>

namespace {

using ov::test::InverseLayerTest;

std::vector<float> data_4x4_f32 =
    {7.0f, -2.0f, 5.0f, 8.0f, -6.0f, 3.0f, -2.0f, 27.0f, 10.0f, -12.0f, 23.0f, 21.0f, 1.0f, -21.0f, 16.0f, 15.0f};

std::vector<ov::float16> data_2x3x3_f16 = {ov::float16(2.0f),
                                           ov::float16(-1.0f),
                                           ov::float16(0.0f),
                                           ov::float16(-1.0f),
                                           ov::float16(2.0f),
                                           ov::float16(-1.0f),
                                           ov::float16(0.0f),
                                           ov::float16(-1.0f),
                                           ov::float16(2.0f),

                                           ov::float16(3.0f),
                                           ov::float16(1.0f),
                                           ov::float16(2.0f),
                                           ov::float16(0.0f),
                                           ov::float16(4.0f),
                                           ov::float16(1.0f),
                                           ov::float16(2.0f),
                                           ov::float16(-2.0f),
                                           ov::float16(0.0f)};

std::vector<ov::bfloat16> data_2x2_bf16 = {ov::bfloat16(0.1f),
                                           ov::bfloat16(1.0f),
                                           ov::bfloat16(10.0f),
                                           ov::bfloat16(101.0f)};

const auto data = testing::Values(ov::Tensor(ov::element::f32, {4, 4}, data_4x4_f32.data()),
                                  ov::Tensor(ov::element::f16, {2, 3, 3}, data_2x3x3_f16.data()),
                                  ov::Tensor(ov::element::bf16, {2, 2}, data_2x2_bf16.data()));

const auto adjoint = testing::Values(true, false);

const auto test_type_static = testing::Values("static");
const auto test_type_dynamic = testing::Values("dynamic");

const auto device_cpu = testing::Values(ov::test::utils::DEVICE_CPU);

const auto params_static = ::testing::Combine(test_type_static, data, adjoint, device_cpu);

const auto params_dynamic = ::testing::Combine(test_type_dynamic, data, adjoint, device_cpu);

INSTANTIATE_TEST_SUITE_P(smoke_InverseStatic, InverseLayerTest, params_static, InverseLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_InverseDynamic, InverseLayerTest, params_dynamic, InverseLayerTest::getTestCaseName);
}  // namespace
