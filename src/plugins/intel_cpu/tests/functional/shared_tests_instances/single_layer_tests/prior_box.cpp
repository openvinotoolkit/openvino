// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_op_tests/prior_box.hpp"
#include "common_test_utils/test_constants.hpp"

namespace {
using ov::test::PriorBoxLayerTest;

const std::vector<ov::element::Type> model_types = {
    ov::element::i32,
    ov::element::u16};

const std::vector<std::vector<float>> min_sizes = {
    {256.0f}};

const std::vector<std::vector<float>> max_sizes = {
    {315.0f}};

const std::vector<std::vector<float>> aspect_ratios = {
    {2.0f}};

const std::vector<std::vector<float>> densities = {
    {1.0f}};

const std::vector<std::vector<float>> fixed_ratios = {
    {}};

const std::vector<std::vector<float>> fixed_sizes = {
    {}};

const std::vector<bool> clips = {
    false, true};

const std::vector<bool> flips = {
    false, true};

const std::vector<float> steps = {
    1.0f,
};

const std::vector<float> offsets = {
    0.0f,
};

const std::vector<std::vector<float>> variances = {
    {}};

const std::vector<bool> scale_all_sizes = {
    false, true};

const std::vector<bool> min_max_aspect_ratios_order = {
    false, true};

const std::vector<ov::Shape> input_shapes_static = {{300, 300}, {32, 32}};

const auto layerSpecificParams = ::testing::Combine(
    ::testing::ValuesIn(min_sizes),
    ::testing::ValuesIn(max_sizes),
    ::testing::ValuesIn(aspect_ratios),
    ::testing::ValuesIn(densities),
    ::testing::ValuesIn(fixed_ratios),
    ::testing::ValuesIn(fixed_sizes),
    ::testing::ValuesIn(clips),
    ::testing::ValuesIn(flips),
    ::testing::ValuesIn(steps),
    ::testing::ValuesIn(offsets),
    ::testing::ValuesIn(variances),
    ::testing::ValuesIn(scale_all_sizes),
    ::testing::ValuesIn(min_max_aspect_ratios_order));

INSTANTIATE_TEST_SUITE_P(smoke_PriorBox_Basic, PriorBoxLayerTest,
                            ::testing::Combine(
                                layerSpecificParams,
                                ::testing::ValuesIn(model_types),
                                ::testing::Values(ov::test::static_shapes_to_test_representation(input_shapes_static)),
                                ::testing::Values(ov::test::utils::DEVICE_CPU)),
                            PriorBoxLayerTest::getTestCaseName);
} // namespace
