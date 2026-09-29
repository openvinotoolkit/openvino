// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/convert_color_to_nv12.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "gtest/gtest.h"

namespace {
using ov::test::ConvertColorToNV12LayerTest;

const std::vector<std::vector<ov::Shape>> in_shapes = {{{1, 10, 10, 3}}};

const std::vector<ov::element::Type> inTypes = {ov::element::u8, ov::element::f32};

const std::vector<std::vector<ov::Shape>> in_shapes_acc = {{{1, 16 * 6, 16, 3}}};

const std::vector<std::vector<ov::Shape>> in_shapes_nightly = {{{1, 256 * 256, 256, 3}}};

const auto test_case_values =
    ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(in_shapes)),
                       ::testing::ValuesIn(inTypes),
                       ::testing::Bool(),
                       ::testing::Bool(),
                       ::testing::Values(ov::test::utils::DEVICE_CPU));

INSTANTIATE_TEST_SUITE_P(smoke_TestsConvertColorToNV12,
                         ConvertColorToNV12LayerTest,
                         test_case_values,
                         ConvertColorToNV12LayerTest::getTestCaseName);

const auto testCase_accuracy_values =
    ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(in_shapes_acc)),
                       ::testing::Values(ov::element::u8),
                       ::testing::Bool(),
                       ::testing::Bool(),
                       ::testing::Values(ov::test::utils::DEVICE_CPU));

INSTANTIATE_TEST_SUITE_P(smoke_TestsConvertColorToNV12_acc,
                         ConvertColorToNV12LayerTest,
                         testCase_accuracy_values,
                         ConvertColorToNV12LayerTest::getTestCaseName);

const auto testCase_accuracy_values_nightly =
    ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(in_shapes_nightly)),
                       ::testing::Values(ov::element::u8),
                       ::testing::Bool(),
                       ::testing::Bool(),
                       ::testing::Values(ov::test::utils::DEVICE_CPU));

INSTANTIATE_TEST_SUITE_P(nightly_TestsConvertColorToNV12_acc,
                         ConvertColorToNV12LayerTest,
                         testCase_accuracy_values_nightly,
                         ConvertColorToNV12LayerTest::getTestCaseName);

}  // namespace
