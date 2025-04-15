// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/multinomial.hpp"

#include <openvino/core/type/element_type.hpp>
#include <openvino/runtime/tensor.hpp>

namespace {

using ov::test::MultinomialLayerTest;

std::vector<float> probs_4x4_f32 = {0.00001f,
                                    0.001f,
                                    0.1f,
                                    10.0f,
                                    0.001f,
                                    0.00001f,
                                    10.0f,
                                    0.1f,
                                    0.1f,
                                    10.0f,
                                    0.00001f,
                                    0.001f,
                                    10.0f,
                                    0.1f,
                                    0.001f,
                                    0.00001f};

std::vector<ov::float16> probs_2x3_f16 = {ov::float16(0.001f),
                                          ov::float16(0.1f),
                                          ov::float16(10.0f),
                                          ov::float16(10.0f),
                                          ov::float16(0.001f),
                                          ov::float16(0.1f)};

std::vector<ov::bfloat16> probs_1x3_bf16 = {ov::bfloat16(0.1f), ov::bfloat16(1.0f), ov::bfloat16(10.0f)};

std::vector<float> probs_4x4_f32_log =
    {3.0f, 6.0f, 10.0f, 0.0f, 3.0f, 0.0f, 10.0f, 6.0f, 6.0f, 10.0f, 0.0f, 3.0f, 10.0f, 6.0f, 3.0f, 0.0f};

std::vector<ov::float16> probs_2x3_f16_log = {ov::float16(3.0f),
                                              ov::float16(6.0f),
                                              ov::float16(10.0f),
                                              ov::float16(10.0f),
                                              ov::float16(3.0f),
                                              ov::float16(6.0f)};

std::vector<ov::bfloat16> probs_1x3_bf16_log = {ov::bfloat16(3.0f), ov::bfloat16(6.0f), ov::bfloat16(10.0f)};

std::vector<int> num_samples_scalar_i32 = {1};
std::vector<int64_t> num_samples_1x1_i64 = {2};

const auto probs = testing::Values(ov::Tensor(ov::element::f32, {4, 4}, probs_4x4_f32.data()),
                                   ov::Tensor(ov::element::f16, {2, 3}, probs_2x3_f16.data()),
                                   ov::Tensor(ov::element::bf16, {1, 3}, probs_1x3_bf16.data()));

const auto probs_log = testing::Values(ov::Tensor(ov::element::f32, {4, 4}, probs_4x4_f32_log.data()),
                                       ov::Tensor(ov::element::f16, {2, 3}, probs_2x3_f16_log.data()),
                                       ov::Tensor(ov::element::bf16, {1, 3}, probs_1x3_bf16_log.data()));

const auto num_samples = testing::Values(ov::Tensor(ov::element::i32, {}, num_samples_scalar_i32.data()),
                                         ov::Tensor(ov::element::i64, {1}, num_samples_1x1_i64.data()));

const auto convert_type = testing::Values(ov::test::ElementType::i32, ov::test::ElementType::i64);

const auto with_replacement = testing::Values(true, false);

const auto log_probs_true = testing::Values(true);
const auto log_probs_false = testing::Values(false);

const auto test_type_static = testing::Values("static");
const auto test_type_dynamic = testing::Values("dynamic");

// NOTE:  (0,0) seeds are skipped (ticket 126095)
const auto global_op_seed =
    testing::Values(std::pair<uint64_t, uint64_t>{1ul, 2ul}, std::pair<uint64_t, uint64_t>{0ul, 0ul});

const auto device_cpu = testing::Values(ov::test::utils::DEVICE_CPU);

const auto params_static = ::testing::Combine(test_type_static,
                                              probs,
                                              num_samples,
                                              convert_type,
                                              with_replacement,
                                              log_probs_false,
                                              global_op_seed,
                                              device_cpu);

const auto params_static_log = ::testing::Combine(test_type_static,
                                                  probs_log,
                                                  num_samples,
                                                  convert_type,
                                                  with_replacement,
                                                  log_probs_true,
                                                  global_op_seed,
                                                  device_cpu);

const auto params_dynamic = ::testing::Combine(test_type_dynamic,
                                               probs,
                                               num_samples,
                                               convert_type,
                                               with_replacement,
                                               log_probs_false,
                                               global_op_seed,
                                               device_cpu);

const auto params_dynamic_log = ::testing::Combine(test_type_dynamic,
                                                   probs_log,
                                                   num_samples,
                                                   convert_type,
                                                   with_replacement,
                                                   log_probs_true,
                                                   global_op_seed,
                                                   device_cpu);

INSTANTIATE_TEST_SUITE_P(smoke_MultinomialStatic,
                         MultinomialLayerTest,
                         params_static,
                         MultinomialLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MultinomialStaticLog,
                         MultinomialLayerTest,
                         params_static_log,
                         MultinomialLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MultinomialDynamic,
                         MultinomialLayerTest,
                         params_dynamic,
                         MultinomialLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MultinomialDynamicLog,
                         MultinomialLayerTest,
                         params_dynamic_log,
                         MultinomialLayerTest::getTestCaseName);
}  // namespace
