// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/reshape.hpp"
#include "common_test_utils/test_constants.hpp"

namespace {
using ov::test::ReshapeLayerTest;
using ov::test::reshapeParams;

const std::vector<ov::element::Type> netPrecisions = {
            ov::element::f32,
            ov::element::f16,
            ov::element::i64
};

INSTANTIATE_TEST_SUITE_P(smoke_ReshapeCheck, ReshapeLayerTest,
        ::testing::Combine(
                ::testing::Values(true),
                ::testing::ValuesIn(netPrecisions),
                ::testing::Values(std::vector<size_t>({10, 10, 10, 10})),
                ::testing::Values(std::vector<int64_t>({10, 0, 100})),
                ::testing::Values(ov::test::utils::DEVICE_GPU)),
                ReshapeLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ReshapeCheckNegative, ReshapeLayerTest,
        ::testing::Combine(
                ::testing::Values(true),
                ::testing::ValuesIn(netPrecisions),
                ::testing::Values(std::vector<size_t>({10, 10, 10, 10})),
                ::testing::Values(std::vector<int64_t>({10, -1, 100})),
                ::testing::Values(ov::test::utils::DEVICE_GPU)),
                ReshapeLayerTest::getTestCaseName);

static std::vector<reshapeParams> generate_tests() {
    std::vector<reshapeParams> res;
    using TargetPattern = std::vector<int64_t>;
    using InputShape = std::vector<size_t>;
    std::vector<std::pair<InputShape, TargetPattern>> params = {
        {{1, 2, 3, 4}, {-1}},
        {{1, 2, 3, 4}, {1, 6, 4}},
        {{1, 3, 4}, {1, 3, 2, 2}},
        {{2, 2, 3, 4, 5}, {4, 3, 20}},
        {{2, 2, 3, 4, 5}, {12, 2, 2, 5}},
        {{2, 2, 3, 4, 5}, {2, 2, 3, 1, 4, 5}},
        {{2, 2, 3, 1, 4, 5}, {2, 2, 3, 4, 5}},
        {{2, 2, 3, 1, 4, 5}, {2, 1, -1, 5}},
        {{2, 2, 3, 1, 4, 5}, {2, 1, -1}},
        {{2, 2, 3, 1, 4, 5}, {2, -1}},
        {{2, 2, 3, 1, 6, 8}, {2, 2, 3, 1, 3, 2, 4, 2}},
        {{2, 2, 3, 1, 3, 2, 4, 2}, {2, 2, 3, 1, 6, 8}},
    };
    for (auto& p : params) {
        reshapeParams test_case = std::make_tuple(false, ov::element::f16,
                                      p.first, p.second, ov::test::utils::DEVICE_GPU);
        res.push_back(test_case);
    }

    return res;
}

INSTANTIATE_TEST_SUITE_P(smoke_ReshapeTests, ReshapeLayerTest, ::testing::ValuesIn(generate_tests()), ReshapeLayerTest::getTestCaseName);

}  // namespace
