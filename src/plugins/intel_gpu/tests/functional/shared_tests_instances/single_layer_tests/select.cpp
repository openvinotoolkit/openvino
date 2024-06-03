// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "single_op_tests/select.hpp"
#include "common_test_utils/test_constants.hpp"

using ov::test::SelectLayerTest;

const std::vector<ov::element::Type> inputPrecision = {
    ov::element::u8,
    ov::element::f16,
    ov::element::f32,
    ov::element::i16,
    ov::element::i32
};

const std::vector<std::vector<ov::Shape>> noneShapes = {
    {{1}, {1}, {1}},
    {{8}, {8}, {8}},
    {{4, 5}, {4, 5}, {4, 5}},
    {{3, 4, 5}, {3, 4, 5}, {3, 4, 5}},
    {{2, 3, 4, 5}, {2, 3, 4, 5}, {2, 3, 4, 5}},
    {{2, 2, 2, 2, 2}, {2, 2, 2, 2, 2}, {2, 2, 2, 2, 2}},
};

const std::vector<std::vector<ov::Shape>> numpyShapes = {
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
    {{1, 1, 3}, {1, 3, 1}, {3, 1, 1}},
    {{2, 2, 2, 2, 2}, {2, 2, 2, 2}, {2, 2, 2, 2, 2}},
    {{2, 2, 2, 2, 2}, {2, 2, 2, 2, 2}, {2, 2, 2, 2}}
};

INSTANTIATE_TEST_SUITE_P(smoke_CLDNN_TestsSelect_none,
                         SelectLayerTest,
                         ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(noneShapes)),
                                            ::testing::ValuesIn(inputPrecision),
                                            ::testing::Values(ov::op::AutoBroadcastType::NONE),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         SelectLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_CLDNN_TestsSelect_numpy,
                         SelectLayerTest,
                         ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(numpyShapes)),
                                            ::testing::ValuesIn(inputPrecision),
                                            ::testing::Values(ov::op::AutoBroadcastType::NUMPY),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         SelectLayerTest::getTestCaseName);
