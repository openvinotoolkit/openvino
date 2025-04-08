// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_op_tests/one_hot.hpp"

namespace {
using ov::test::OneHotLayerTest;

const std::vector<ov::element::Type> model_types = {
        ov::element::i32,
};

const std::vector<ov::element::Type> arg_depth_type_ic = { ov::element::i32 };
const std::vector<int64_t> arg_depth_ic = { 1, 5, 1017 };
const std::vector<ov::element::Type> arg_set_type_ic = { ov::element::i32 };
const std::vector<float> arg_on_value_ic = { 0, 1, -29 };
const std::vector<float> arg_off_value_ic = { 0, 1, -127 };
const std::vector<int64_t> arg_axis_ic = {0};
const std::vector<std::vector<ov::Shape>> input_shapes_ic = {{{4, 5}}, {{3, 7}}};

const auto oneHotParams_ic = testing::Combine(
        testing::ValuesIn(arg_depth_type_ic),
        testing::ValuesIn(arg_depth_ic),
        testing::ValuesIn(arg_set_type_ic),
        testing::ValuesIn(arg_on_value_ic),
        testing::ValuesIn(arg_off_value_ic),
        testing::ValuesIn(arg_axis_ic),
        testing::ValuesIn(model_types),
        testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shapes_ic)),
        testing::Values(ov::test::utils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(
        smoke_OneHotIntConst,
        OneHotLayerTest,
        oneHotParams_ic,
        OneHotLayerTest::getTestCaseName
);


const std::vector<ov::element::Type> arg_depth_type_ax = { ov::element::i32 };
const std::vector<int64_t> arg_depth_ax = { 3 };
const std::vector<ov::element::Type> arg_set_type_ax = { ov::element::i32, ov::element::f32 };
const std::vector<float> arg_on_value_ax = { 17 };
const std::vector<float> arg_off_value_ax = { -3 };
const std::vector<int64_t> arg_axis_ax = {0, 1, 3, 5, -4, -5};
const std::vector<std::vector<ov::Shape>> input_shapes_ax = {{{4, 8, 5, 3, 2, 9}}};

const auto oneHotParams_ax = testing::Combine(
        testing::ValuesIn(arg_depth_type_ax),
        testing::ValuesIn(arg_depth_ax),
        testing::ValuesIn(arg_set_type_ax),
        testing::ValuesIn(arg_on_value_ax),
        testing::ValuesIn(arg_off_value_ax),
        testing::ValuesIn(arg_axis_ax),
        testing::ValuesIn(model_types),
        testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shapes_ax)),
        testing::Values(ov::test::utils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(
        smoke_OneHotAxrng,
        OneHotLayerTest,
        oneHotParams_ax,
        OneHotLayerTest::getTestCaseName
);


const std::vector<ov::element::Type> arg_depth_type_t = { ov::element::i8, ov::element::u8 };
const std::vector<int64_t> arg_depth_t = { 1 };
const std::vector<ov::element::Type> arg_set_type_t = { ov::element::i8, ov::element::u8,
                                                          ov::element::bf16, ov::element::f32 };
const std::vector<float> arg_on_value_t = { 1 };
const std::vector<float> arg_off_value_t = { 1 };
const std::vector<int64_t> arg_axis_t = {-1};
const std::vector<std::vector<ov::Shape>> input_shapes_t = {{{2, 2}}};

const auto oneHotParams_t = testing::Combine(
        testing::ValuesIn(arg_depth_type_t),
        testing::ValuesIn(arg_depth_t),
        testing::ValuesIn(arg_set_type_t),
        testing::ValuesIn(arg_on_value_t),
        testing::ValuesIn(arg_off_value_t),
        testing::ValuesIn(arg_axis_t),
        testing::ValuesIn(model_types),
        testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shapes_t)),
        testing::Values(ov::test::utils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(
        smoke_OneHotArgType,
        OneHotLayerTest,
        oneHotParams_t,
        OneHotLayerTest::getTestCaseName
);
}  // namespace
