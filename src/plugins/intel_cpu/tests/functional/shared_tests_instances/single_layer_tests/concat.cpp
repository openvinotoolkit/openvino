// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_op_tests/concat.hpp"
#include "common_test_utils/test_constants.hpp"


namespace {
using ov::test::ConcatLayerTest;
using ov::test::ConcatStringLayerTest;

std::vector<int> axes = {-3, -2, -1, 0, 1, 2, 3};
std::vector<std::vector<ov::Shape>> shapes_static = {
        {{10, 10, 10, 10}},
        {{10, 10, 10, 10}, {10, 10, 10, 10}},
        {{10, 10, 10, 10}, {10, 10, 10, 10}, {10, 10, 10, 10}},
        {{10, 10, 10, 10}, {10, 10, 10, 10}, {10, 10, 10, 10}, {10, 10, 10, 10}},
        {{10, 10, 10, 10}, {10, 10, 10, 10}, {10, 10, 10, 10}, {10, 10, 10, 10}, {10, 10, 10, 10}}
};


std::vector<ov::element::Type> netPrecisions = {ov::element::f32,
                                                ov::element::f16};

INSTANTIATE_TEST_SUITE_P(smoke_NoReshape, ConcatLayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(axes),
                                ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(shapes_static)),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        ConcatLayerTest::getTestCaseName);

std::vector<ov::test::ConcatStringParamsTuple> stringConcatParams{
    {0, std::vector<ov::Shape>{{2}, {1}}, ov::element::string, ov::test::utils::DEVICE_CPU, {{"   ", "..."}, {"abc"}}},
    {0,
     std::vector<ov::Shape>{{2, 1}, {2, 1}},
     ov::element::string,
     ov::test::utils::DEVICE_CPU,
     {{"   ", "..."}, {"abc", "12"}}},
    {1,
     std::vector<ov::Shape>{{1, 2}, {1, 2}},
     ov::element::string,
     ov::test::utils::DEVICE_CPU,
     {{"   ", "..."}, {"abc", "12"}}},
    {2,
     std::vector<ov::Shape>{{1, 1, 2, 1}, {1, 1, 3, 1}, {1, 1, 4, 1}},
     ov::element::string,
     ov::test::utils::DEVICE_CPU,
     {{"   ", "..."}, {" Abc", "1 2", ""}, {"dEfg ", "3 4", " /0", "/n"}}},
    {1,
     std::vector<ov::Shape>{{2, 2}, {2, 3}, {2, 4}},
     ov::element::string,
     ov::test::utils::DEVICE_CPU,
     {{"   ", "...", "   ", "..."},
      {"a", ",b", "c. ", "d d ", " e ", "F"},
      {"defg ", ".", "3 4", ":", " /0", "_", "/n", ";"}}},
    {3,
     std::vector<ov::Shape>{{1, 2, 1, 2}, {1, 2, 1, 3}, {1, 2, 1, 4}},
     ov::element::string,
     ov::test::utils::DEVICE_CPU,
     {{"   ", "...", "   ", "..."},
      {"a", ",b", "c. ", "d d ", " e ", "F"},
      {"defg ", ".", "3 4", ":", " /0", "_", "/n", ";"}}}};

INSTANTIATE_TEST_SUITE_P(smoke_concat_string,
                         ConcatStringLayerTest,
                         ::testing::ValuesIn(stringConcatParams),
                         ConcatStringLayerTest::getTestCaseName);
}  // namespace
