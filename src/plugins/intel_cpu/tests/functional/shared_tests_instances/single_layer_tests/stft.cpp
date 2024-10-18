// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/stft.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"

namespace ov {
namespace test {
using ov::test::STFTLayerTest;

const std::vector<ov::element::Type> data_type = {ov::element::f32, ov::element::bf16};
const std::vector<ov::element::Type> step_size_type = {ov::element::i32, ov::element::i64};

const std::vector<std::vector<InputShape>> input_shapes = {
    {{{}, {{1, 128}}}, {{}, {{8}}}, {{}, {{}}}, {{}, {{}}}},
    {{{}, {{2, 226}}}, {{}, {{16}}}, {{}, {{}}}, {{}, {{}}}},
    {{{-1, -1}, {{1, 128}}}, {{}, {{8}}}, {{}, {{}}}, {{}, {{}}}},
    {{{{2, 4}, {1, 300}}, {{2, 226}}}, {{-1}, {{16}}}, {{}, {{}}}, {{}, {{}}}},
};

const std::vector<int64_t> frame_size = {16, 24};
const std::vector<int64_t> step_size = {2, 3, 4};

const std::vector<bool> transpose_frames = {
    false,
    true,
};

std::vector<utils::InputLayerType> in_types = {utils::InputLayerType::CONSTANT, utils::InputLayerType::PARAMETER};

const auto testCaseStatic = ::testing::Combine(::testing::ValuesIn(input_shapes),
                                               ::testing::ValuesIn(frame_size),
                                               ::testing::ValuesIn(step_size),
                                               ::testing::ValuesIn(transpose_frames),
                                               ::testing::ValuesIn(data_type),
                                               ::testing::ValuesIn(step_size_type),
                                               ::testing::ValuesIn(in_types),
                                               ::testing::Values(ov::test::utils::DEVICE_CPU));

INSTANTIATE_TEST_SUITE_P(smoke_STFT_static, STFTLayerTest, testCaseStatic, STFTLayerTest::getTestCaseName);
}  // namespace test
}  // namespace ov
