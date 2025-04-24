// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_op_tests/select.hpp"
#include "common_test_utils/test_constants.hpp"

using ov::test::SelectLayerTest;

const std::vector<ov::element::Type> model_types = {
    ov::element::i8,
    ov::element::i16,
    ov::element::i32,
    ov::element::f32
};

const std::vector<std::vector<ov::Shape>> none_shapes = {
    {{1}, {1}, {1}},
    {{8}, {8}, {8}},
    {{4, 5}, {4, 5}, {4, 5}},
    {{3, 4, 5}, {3, 4, 5}, {3, 4, 5}},
    {{2, 3, 4, 5}, {2, 3, 4, 5}, {2, 3, 4, 5}},
    {{2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}}
};

const auto none_cases = ::testing::Combine(
    ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(none_shapes)),
    ::testing::ValuesIn(model_types),
    ::testing::Values(ov::op::AutoBroadcastType::NONE),
    ::testing::Values(ov::test::utils::DEVICE_CPU)
);

const std::vector<std::vector<ov::Shape>> numpy_shapes = {
    {{1}, {1}, {1}},
    {{1}, {16}, {1}},
    {{1}, {1}, {16}},
    {{1}, {8}, {8}},
    {{8}, {1}, {8}},
    {{8}, {8}, {8}},
    {{4, 1}, {1}, {4, 8}},
    {{3, 8}, {8}, {3, 1}},
    {{8, 1}, {8, 1}, {8, 1}},
    {{1}, {5, 8}, {5, 8}},
    {{8, 1, 1}, {8, 1, 1}, {2, 5}},
    {{8, 1}, {6, 8, 1}, {6, 1, 1}},
    {{5, 1}, {8, 1, 7}, {5, 7}},
    {{2, 8, 1}, {2, 8, 9}, {2, 1, 9}},
    {{1, 4}, {8, 1, 1, 1}, {4}},
    {{5, 4, 1}, {8, 5, 1, 1}, {4, 1}},
    {{1, 4}, {6, 1, 8, 1}, {6, 1, 8, 4}},
    {{7, 3, 1, 8}, {7, 1, 1, 8}, {3, 2, 8}},
    {{1, 3, 1}, {8, 2, 3, 1}, {3, 9}},
    {{5, 1, 8}, {2, 1, 9, 8}, {2, 5, 9, 8}},
    {{6, 1, 1, 8}, {6, 7, 1, 8}, {2, 1}},
    {{5, 1, 1, 1}, {5, 7, 8, 6}, {1, 8, 6}},
    {{8, 1, 5}, {8, 1, 1, 1, 1}, {8, 7, 5}},
    {{8, 1, 1, 9}, {4, 8, 1, 1, 1}, {1, 1, 9}},
    {{5, 1, 2, 1}, {8, 1, 9, 1, 1}, {5, 1, 2, 1}},
    {{8, 1}, {2, 1, 1, 8, 1}, {9, 1, 1}},
    {{8, 5, 5, 5, 1}, {8, 1, 1, 1, 8}, {5, 5, 5, 8}},
    {{4}, {8, 5, 6, 1, 1}, {2, 4}},
    {{9, 9, 2, 8, 1}, {9, 1, 2, 8, 1}, {9, 1, 1, 1}},
    {{5, 3, 3}, {8, 1, 1, 3, 3}, {5, 1, 3}},
    {{5, 1, 8, 1}, {5, 5, 1, 8, 1}, {1}},
    {{3}, {6, 8, 1, 1, 3}, {6, 1, 5, 3, 3}},
    {{5, 1}, {3, 1, 4, 1, 8}, {1, 4, 5, 8}},
    {{2, 1, 5}, {8, 6, 2, 3, 1}, {5}},
    {{6}, {2, 1, 9, 8, 6}, {2, 4, 9, 8, 6}},
    {{5, 7, 1, 8, 1}, {5, 7, 1, 8, 4}, {8, 1}},
    {{7, 6, 5, 8}, {4, 7, 6, 5, 8}, {6, 1, 8}}
};

const auto numpy_cases = ::testing::Combine(
    ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(numpy_shapes)),
    ::testing::ValuesIn(model_types),
    ::testing::Values(ov::op::AutoBroadcastType::NUMPY),
    ::testing::Values(ov::test::utils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_TestsSelect_none, SelectLayerTest, none_cases, SelectLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_TestsSelect_numpy, SelectLayerTest, numpy_cases, SelectLayerTest::getTestCaseName);
