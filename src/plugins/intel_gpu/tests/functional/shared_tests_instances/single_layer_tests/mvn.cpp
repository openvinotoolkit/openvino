// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_op_tests/mvn.hpp"
#include "common_test_utils/test_constants.hpp"

namespace {
using ov::test::Mvn1LayerTest;
using ov::test::Mvn6LayerTest;

const std::vector<ov::AxisSet> emptyReductionAxes = {{}};

const std::vector<std::vector<ov::Shape>> inputShapes = {
    {{1, 32, 17}},
    {{1, 37, 9}},
    {{1, 16, 5, 8}},
    {{2, 19, 5, 10}},
    {{7, 32, 2, 8}},
    {{5, 8, 3, 5}},
    {{4, 41, 6, 9}},
    {{1, 32, 8, 1, 6}},
    {{1, 9, 1, 15, 9}},
    {{6, 64, 6, 1, 18}},
    {{2, 31, 2, 9, 1}},
    {{10, 16, 5, 10, 6}}
};

const std::vector<bool> acrossChannels = {
    true,
    false
};

const std::vector<bool> normalizeVariance = {
    true,
    false
};

const std::vector<double> epsilon = {
    0.000000001
};

INSTANTIATE_TEST_SUITE_P(smoke_CLDNN_TestsMVN,
                         Mvn1LayerTest,
                         ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapes)),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::ValuesIn(emptyReductionAxes),
                                            ::testing::ValuesIn(acrossChannels),
                                            ::testing::ValuesIn(normalizeVariance),
                                            ::testing::ValuesIn(epsilon),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         Mvn1LayerTest::getTestCaseName);

std::vector<ov::element::Type> dataPrecisions = {
    ov::element::f32,
    ov::element::f16
};

std::vector<ov::element::Type> idxPrecisions = {
    ov::element::i32,
    ov::element::i64
};

const std::vector<std::string> epsMode = {
    "inside_sqrt",
    "outside_sqrt"
};

const std::vector<float> epsilonF = {
    0.0001f
};

INSTANTIATE_TEST_SUITE_P(smoke_MVN_5D, Mvn6LayerTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(
                                std::vector<std::vector<ov::Shape>>{{{1, 10, 5, 7, 8}}, {{1, 3, 8, 9, 49}}})),
                            ::testing::ValuesIn(dataPrecisions),
                            ::testing::ValuesIn(idxPrecisions),
                            ::testing::ValuesIn(std::vector<std::vector<int>>{{1, 2, 3, 4}, {2, 3, 4}, {-3, -2, -1}, {-1, -4, -2, -3}, {-1}}),
                            ::testing::ValuesIn(normalizeVariance),
                            ::testing::ValuesIn(epsilonF),
                            ::testing::ValuesIn(epsMode),
                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        Mvn6LayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MVN_4D, Mvn6LayerTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(
                                std::vector<std::vector<ov::Shape>>{{{1, 10, 5, 17}}, {{1, 3, 8, 9}}})),
                            ::testing::ValuesIn(dataPrecisions),
                            ::testing::ValuesIn(idxPrecisions),
                            ::testing::ValuesIn(std::vector<std::vector<int>>{{1, 2, 3}, {2, 3}, {-2, -1}, {-2, -1, -3}, {-1}}),
                            ::testing::ValuesIn(normalizeVariance),
                            ::testing::ValuesIn(epsilonF),
                            ::testing::ValuesIn(epsMode),
                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        Mvn6LayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MVN_3D, Mvn6LayerTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(
                                std::vector<std::vector<ov::Shape>>{{{1, 32, 17}}, {{1, 37, 9}}})),
                            ::testing::ValuesIn(dataPrecisions),
                            ::testing::ValuesIn(idxPrecisions),
                            ::testing::ValuesIn(std::vector<std::vector<int>>{{1, 2}, {2}, {-1}, {-1, -2}}),
                            ::testing::ValuesIn(normalizeVariance),
                            ::testing::ValuesIn(epsilonF),
                            ::testing::ValuesIn(epsMode),
                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        Mvn6LayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MVN_2D, Mvn6LayerTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(
                                std::vector<std::vector<ov::Shape>>{{{3, 5}}, {{2, 55}}})),
                            ::testing::ValuesIn(dataPrecisions),
                            ::testing::ValuesIn(idxPrecisions),
                            ::testing::ValuesIn(std::vector<std::vector<int>>{{1}}),
                            ::testing::ValuesIn(normalizeVariance),
                            ::testing::ValuesIn(epsilonF),
                            ::testing::ValuesIn(epsMode),
                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        Mvn6LayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Decomposition_1D, Mvn6LayerTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(
                                std::vector<std::vector<ov::Shape>>{{{3}}, {{9}}, {{55}}})),
                            ::testing::ValuesIn(dataPrecisions),
                            ::testing::ValuesIn(idxPrecisions),
                            ::testing::ValuesIn(std::vector<std::vector<int>>{{}}),
                            ::testing::ValuesIn(normalizeVariance),
                            ::testing::ValuesIn(epsilonF),
                            ::testing::ValuesIn(epsMode),
                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        Mvn6LayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Decomposition_3D, Mvn6LayerTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(
                                std::vector<std::vector<ov::Shape>>{{{1, 32, 17}}, {{1, 37, 9}}})),
                            ::testing::ValuesIn(dataPrecisions),
                            ::testing::ValuesIn(idxPrecisions),
                            ::testing::ValuesIn(std::vector<std::vector<int>>{{0, 1, 2}, {0}, {1}}),
                            ::testing::ValuesIn(normalizeVariance),
                            ::testing::ValuesIn(epsilonF),
                            ::testing::ValuesIn(epsMode),
                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        Mvn6LayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Decomposition_4D, Mvn6LayerTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(
                                std::vector<std::vector<ov::Shape>>{{{1, 16, 5, 8}}, {{2, 19, 5, 10}}})),
                            ::testing::ValuesIn(dataPrecisions),
                            ::testing::ValuesIn(idxPrecisions),
                            ::testing::ValuesIn(std::vector<std::vector<int>>{{0, 1, 2, 3}, {0, 1, 2}, {0, 3}, {0}, {1}, {2}, {3}}),
                            ::testing::ValuesIn(normalizeVariance),
                            ::testing::ValuesIn(epsilonF),
                            ::testing::ValuesIn(epsMode),
                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        Mvn6LayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Decomposition_6D, Mvn6LayerTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(
                                std::vector<std::vector<ov::Shape>>{{{1, 3, 5, 4, 2, 6}}})),
                            ::testing::ValuesIn(dataPrecisions),
                            ::testing::ValuesIn(idxPrecisions),
                            ::testing::ValuesIn(std::vector<std::vector<int>>{{0, 1, 5}, {0, 1, 2, 3}, {0, 1, 2}, {0, 3}, {0}, {3}}),
                            ::testing::ValuesIn(normalizeVariance),
                            ::testing::ValuesIn(epsilonF),
                            ::testing::ValuesIn(epsMode),
                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        Mvn6LayerTest::getTestCaseName);

}  // namespace
