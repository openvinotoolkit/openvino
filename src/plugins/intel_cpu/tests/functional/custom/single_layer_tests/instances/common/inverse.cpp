// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/inverse.hpp"

#include <openvino/core/type/element_type.hpp>
#include <openvino/runtime/tensor.hpp>

namespace {

using ov::test::InverseLayerTest;

std::vector<float> data_4x4_f32 =
    {5.0f, 6.0f, 6.0f, 8.0f, 2.0f, 2.0f, 2.0f, 8.0f, 6.0f, 6.0f, 2.0f, 8.0f, 2.0f, 3.0f, 6.0f, 7.0f};

    // -17 -9 12 16
    // 17 8.75 -11.75 -16
    // -4 -2.25 2.75 4
    // 1 0.75 -0.75 -1

    // 136 72 -96 -128
    // -136 -70 94 128
    // 32 18 -22 -32
    // -8 -6 6 8

std::vector<ov::float16> data_2x3x3_f16 = {ov::float16(2.0f),
                                           ov::float16(-1.0f),
                                           ov::float16(0.0f),
                                           ov::float16(-1.0f),
                                           ov::float16(2.0f),
                                           ov::float16(-1.0f),
                                           ov::float16(0.0f),
                                           ov::float16(-1.0f),
                                           ov::float16(2.0f),

                                           // 0.75 0.5 0.25
                                           // 0.5 1 0.5
                                           // 0.25 0.5 0.75

                                           // 3 2 1
                                           // 2 4 2
                                           // 1 2 3

                                           ov::float16(3.0f),
                                           ov::float16(1.0f),
                                           ov::float16(2.0f),
                                           ov::float16(0.0f),
                                           ov::float16(4.0f),
                                           ov::float16(1.0f),
                                           ov::float16(2.0f),
                                           ov::float16(-2.0f),
                                           ov::float16(0.0f)};

                                            // -0.25 | 0.5 0.875
                                            // -0.25 0.5 0.378
                                            // 1 -1 -1.5

                                            // -2 4 7
                                            // -2 4 3
                                            // 8 -8 -12

std::vector<ov::bfloat16> data_2x2_bf16 = {ov::bfloat16(0.5f),
                                           ov::bfloat16(1.0f),
                                           ov::bfloat16(3.0f),
                                           ov::bfloat16(2.0f)};

                                            // -1 0.5
                                            // 1.5 -0.25

                                            // 2 -1
                                            // -3 0.5

const auto data = testing::Values(ov::Tensor(ov::element::f32, {4, 4}, data_4x4_f32.data()),
                                  ov::Tensor(ov::element::f16, {2, 3, 3}, data_2x3x3_f16.data()),
                                  ov::Tensor(ov::element::bf16, {2, 2}, data_2x2_bf16.data()));

const auto adjoint = testing::Values(false, true);

const auto test_type_static = testing::Values("static");
const auto test_type_dynamic = testing::Values("dynamic");

const auto device_cpu = testing::Values(ov::test::utils::DEVICE_CPU);

const auto params_static = ::testing::Combine(test_type_static, data, adjoint, device_cpu);

const auto params_dynamic = ::testing::Combine(test_type_dynamic, data, adjoint, device_cpu);

INSTANTIATE_TEST_SUITE_P(smoke_InverseStatic, InverseLayerTest, params_static, InverseLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_InverseDynamic, InverseLayerTest, params_dynamic, InverseLayerTest::getTestCaseName);
}  // namespace
