// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/mvn_multiply_add.hpp"

using namespace ov::test;

namespace {

const std::vector<ov::element::Type> netPrecision = {ov::element::f32};

std::vector<ov::element::Type> idxPrecision = {ov::element::i64};

const std::vector<bool> acrossChannels = {true, false};

const std::vector<bool> normalizeVariance = {true, false};

const std::vector<float> epsilon = {0.000000001f};

const std::vector<std::string> epsMode = {"inside_sqrt", "outside_sqrt"};

const std::vector<std::pair<ov::Shape, ov::Shape>> shapes_1D = {
    std::pair<ov::Shape, ov::Shape>{ov::Shape{5}, ov::Shape{5}},
    std::pair<ov::Shape, ov::Shape>{ov::Shape{64}, ov::Shape{64}},
};

const std::vector<std::pair<ov::Shape, ov::Shape>> shapes_2D = {
    std::pair<ov::Shape, ov::Shape>{ov::Shape{1, 5}, ov::Shape{1, 5}},
    std::pair<ov::Shape, ov::Shape>{ov::Shape{2, 17}, ov::Shape{1, 17}},
    std::pair<ov::Shape, ov::Shape>{ov::Shape{9, 64}, ov::Shape{1, 64}},
    std::pair<ov::Shape, ov::Shape>{ov::Shape{5, 15}, ov::Shape{1, 15}},
};

const std::vector<std::pair<ov::Shape, ov::Shape>> shapes_3D = {
    std::pair<ov::Shape, ov::Shape>{ov::Shape{1, 5, 8}, ov::Shape{1, 5, 8}},
    std::pair<ov::Shape, ov::Shape>{ov::Shape{2, 17, 9}, ov::Shape{1, 1, 9}},
    std::pair<ov::Shape, ov::Shape>{ov::Shape{1, 1, 10}, ov::Shape{1, 1, 10}},
    std::pair<ov::Shape, ov::Shape>{ov::Shape{2, 3, 3}, ov::Shape{2, 3, 3}},
};

INSTANTIATE_TEST_SUITE_P(smoke_MVNMultiplyAdd_1D,
                         MVNMultiplyAdd,
                         ::testing::Combine(::testing::ValuesIn(shapes_1D),
                                            ::testing::ValuesIn(netPrecision),
                                            ::testing::ValuesIn(idxPrecision),
                                            ::testing::Values(std::vector<int>{0}),
                                            ::testing::ValuesIn(normalizeVariance),
                                            ::testing::ValuesIn(epsilon),
                                            ::testing::ValuesIn(epsMode),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         MVNMultiplyAdd::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MVNMultiplyAdd_2D,
                         MVNMultiplyAdd,
                         ::testing::Combine(::testing::ValuesIn(shapes_2D),
                                            ::testing::ValuesIn(netPrecision),
                                            ::testing::ValuesIn(idxPrecision),
                                            ::testing::Values(std::vector<int>{1}),
                                            ::testing::ValuesIn(normalizeVariance),
                                            ::testing::ValuesIn(epsilon),
                                            ::testing::ValuesIn(epsMode),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         MVNMultiplyAdd::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MVNMultiplyAdd_3D,
                         MVNMultiplyAdd,
                         ::testing::Combine(::testing::ValuesIn(shapes_3D),
                                            ::testing::ValuesIn(netPrecision),
                                            ::testing::ValuesIn(idxPrecision),
                                            ::testing::Values(std::vector<int>{2}),
                                            ::testing::ValuesIn(normalizeVariance),
                                            ::testing::ValuesIn(epsilon),
                                            ::testing::ValuesIn(epsMode),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         MVNMultiplyAdd::getTestCaseName);

}  // namespace
