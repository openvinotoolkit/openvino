// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/parameter_result.hpp"

#include "common_test_utils/test_constants.hpp"

using namespace ov::test;

namespace {

const std::vector<ov::test::InputShape> inputShapes = {
    ov::test::InputShape{{1, 3, 10, 10}, {{1, 3, 10, 10}, {1, 3, 10, 10}}},
    ov::test::InputShape{{-1, -1, -1, -1}, {{1, 3, 10, 10}, {2, 5, 3, 10}, {1, 3, 10, 10}, {1, 3, 10, 10}}},
    ov::test::InputShape{{{1, 10}, {1, 10}, {1, 10}, {1, 10}},
                         {{1, 3, 10, 10}, {2, 5, 3, 10}, {1, 3, 10, 10}, {1, 3, 10, 10}}},
};

INSTANTIATE_TEST_SUITE_P(smoke_Check,
                         ParameterResultSubgraphTest,
                         ::testing::Combine(::testing::ValuesIn(inputShapes),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         ParameterResultSubgraphTest::getTestCaseName);

}  // namespace
