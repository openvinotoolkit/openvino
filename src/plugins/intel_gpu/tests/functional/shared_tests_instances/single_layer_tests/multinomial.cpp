// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "single_op_tests/multinomial.hpp"

namespace {
using ov::test::MultinomialLayerTest;

const std::vector<ov::test::ElementType> netPrecisions = {
    ov::element::f16,
    ov::element::f32,
};

std::vector<float> probs_1x32_f32_log =
    {3.0f, 6.0f, 10.0f, 0.0f, 3.0f, 0.0f, 10.0f, 6.0f, 6.0f, 10.0f, 0.0f, 3.0f, 10.0f, 6.0f, 3.0f, 0.0f,
     3.0f, 6.0f, 10.0f, 0.0f, 3.0f, 0.0f, 10.0f, 6.0f, 6.0f, 10.0f, 0.0f, 3.0f, 10.0f, 6.0f, 3.0f, 0.0f};

std::vector<float> probs_2x28_f32_log =
    {3.0f, 6.0f, 10.0f, 0.0f, 3.0f, 0.0f, 10.0f, 6.0f, 6.0f, 10.0f, 0.0f, 3.0f, 10.0f, 6.0f,
     10.0f, 0.0f, 3.0f, 0.0f, 10.0f, 6.0f, 6.0f, 10.0f, 0.0f, 3.0f, 10.0f, 6.0f, 3.0f, 0.0f,
     3.0f, 6.0f, 10.0f, 0.0f, 3.0f, 0.0f, 10.0f, 6.0f, 6.0f, 10.0f, 0.0f, 3.0f, 10.0f, 6.0f,
     10.0f, 0.0f, 3.0f, 0.0f, 10.0f, 6.0f, 6.0f, 10.0f, 0.0f, 3.0f, 10.0f, 6.0f, 3.0f, 0.0f};

std::vector<int64_t> num_samples_2_i64 = {2};
std::vector<int64_t> num_samples_4_i64 = {4};

const std::vector<ov::Tensor> inputTensors = {
                    ov::Tensor(ov::element::f32, {1, 32}, probs_1x32_f32_log.data()),
                    ov::Tensor(ov::element::f32, {2, 28}, probs_2x28_f32_log.data())};

const std::vector<ov::Tensor> numSamples = {
                    ov::Tensor(ov::element::i64, {1}, num_samples_2_i64.data()),
                    ov::Tensor(ov::element::i64, {1}, num_samples_4_i64.data())};

const std::vector<bool> withReplacement = {
    false,
    true
};

const std::vector<bool> logProbes = {
    false,
    true
};

INSTANTIATE_TEST_SUITE_P(
    smoke_Multinomial,
    MultinomialLayerTest,
    testing::Combine(testing::Values("static"),
                     testing::ValuesIn(inputTensors),
                     testing::ValuesIn(numSamples),
                     testing::Values(ov::element::i64),
                     testing::ValuesIn(withReplacement),
                     testing::ValuesIn(logProbes),
                     testing::Values(std::pair<uint64_t, uint64_t>{0, 2}),
                     testing::Values(ov::test::utils::DEVICE_GPU)),
                     MultinomialLayerTest::getTestCaseName);
} // anonymous namespace
