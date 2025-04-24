// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_op_tests/mvn.hpp"
#include "common_test_utils/test_constants.hpp"

namespace {
using ov::test::Mvn1LayerTest;
using ov::test::Mvn6LayerTest;

const std::vector<bool> empty_across_channels = {{}};
const std::vector<ov::AxisSet> empty_reduction_axes = {{}};

const std::vector<std::vector<ov::Shape>> input_shapes_static = {
    {{8}},
    {{1, 16}},
    {{3, 19}},
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

const std::vector<bool> across_channels = {
    true,
    false
};

const std::vector<bool> normalize_variance = {
    true,
    false
};

const std::vector<double> epsilon = {
    0.000000001
};

std::vector<ov::element::Type> model_types = {
        ov::element::f16,
        ov::element::f32
};

const auto Mvnacross_channels = ::testing::Combine(
    ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shapes_static)),
    ::testing::ValuesIn(model_types),
    ::testing::ValuesIn(empty_reduction_axes),
    ::testing::ValuesIn(across_channels),
    ::testing::ValuesIn(normalize_variance),
    ::testing::ValuesIn(epsilon),
    ::testing::Values(ov::test::utils::DEVICE_CPU)
);

const std::vector<std::vector<ov::Shape>> input_shapes_reduction_axes_static = {
    {{1, 10, 5, 17}},
    {{1, 3, 8, 9}}
};

const auto MvnReductionAxes = ::testing::Combine(
    ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shapes_reduction_axes_static)),
    ::testing::Values(ov::element::f32),
    ::testing::ValuesIn(std::vector<ov::AxisSet>{{1, 2, 3}, {2, 3}}),
    ::testing::ValuesIn(empty_across_channels),
    ::testing::ValuesIn(normalize_variance),
    ::testing::ValuesIn(epsilon),
    ::testing::Values(ov::test::utils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_TestsMVN_across_channels, Mvn1LayerTest, Mvnacross_channels, Mvn1LayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_TestsMVN_ReductionAxes, Mvn1LayerTest, MvnReductionAxes, Mvn1LayerTest::getTestCaseName);


std::vector<ov::element::Type> idx_types = {
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

const std::vector<std::vector<ov::Shape>> input_shapes_5d_static = {
    {{1, 10, 5, 7, 8}},
    {{1, 3, 8, 9, 49}}
};

INSTANTIATE_TEST_SUITE_P(smoke_MVN_5D, Mvn6LayerTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shapes_5d_static)),
                            ::testing::ValuesIn(model_types),
                            ::testing::ValuesIn(idx_types),
                            ::testing::ValuesIn(std::vector<std::vector<int>>{{1, 2, 3, 4}, {2, 3, 4}, {-3, -2, -1}, {-1, -4, -2, -3}}),
                            ::testing::ValuesIn(normalize_variance),
                            ::testing::ValuesIn(epsilonF),
                            ::testing::ValuesIn(epsMode),
                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        Mvn6LayerTest::getTestCaseName);

const std::vector<std::vector<ov::Shape>> input_shapes_4d_static = {
    {{1, 10, 5, 17}},
    {{1, 3, 8, 9}}
};

INSTANTIATE_TEST_SUITE_P(smoke_MVN_4D, Mvn6LayerTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shapes_4d_static)),
                            ::testing::ValuesIn(model_types),
                            ::testing::ValuesIn(idx_types),
                            ::testing::ValuesIn(std::vector<std::vector<int>>{{1, 2, 3}, {2, 3}, {-2, -1}, {-2, -1, -3}}),
                            ::testing::ValuesIn(normalize_variance),
                            ::testing::ValuesIn(epsilonF),
                            ::testing::ValuesIn(epsMode),
                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        Mvn6LayerTest::getTestCaseName);

const std::vector<std::vector<ov::Shape>> input_shapes_3d_static = {
    {{1, 32, 17}},
    {{1, 37, 9}}
};

INSTANTIATE_TEST_SUITE_P(smoke_MVN_3D, Mvn6LayerTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shapes_3d_static)),
                            ::testing::ValuesIn(model_types),
                            ::testing::ValuesIn(idx_types),
                            ::testing::ValuesIn(std::vector<std::vector<int>>{{1, 2}, {2}, {-1}, {-1, -2}}),
                            ::testing::ValuesIn(normalize_variance),
                            ::testing::ValuesIn(epsilonF),
                            ::testing::ValuesIn(epsMode),
                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        Mvn6LayerTest::getTestCaseName);

const std::vector<std::vector<ov::Shape>> input_shapes_2d_static = {
    {{3, 5}},
    {{2, 55}}
};

INSTANTIATE_TEST_SUITE_P(smoke_MVN_2D, Mvn6LayerTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shapes_2d_static)),
                            ::testing::ValuesIn(model_types),
                            ::testing::ValuesIn(idx_types),
                            ::testing::ValuesIn(std::vector<std::vector<int>>{{1}}),
                            ::testing::ValuesIn(normalize_variance),
                            ::testing::ValuesIn(epsilonF),
                            ::testing::ValuesIn(epsMode),
                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        Mvn6LayerTest::getTestCaseName);

const std::vector<std::vector<ov::Shape>> input_shapes_1d_static = {
    {{3}},
    {{9}},
    {{55}}
};

INSTANTIATE_TEST_SUITE_P(smoke_MVN_1D, Mvn6LayerTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shapes_1d_static)),
                            ::testing::ValuesIn(model_types),
                            ::testing::ValuesIn(idx_types),
                            ::testing::ValuesIn(std::vector<std::vector<int>>{{0}}),
                            ::testing::ValuesIn(normalize_variance),
                            ::testing::ValuesIn(epsilonF),
                            ::testing::ValuesIn(epsMode),
                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        Mvn6LayerTest::getTestCaseName);

const std::vector<std::vector<ov::Shape>> input_shapes_decomposition_3d_static = {
    {{1, 32, 17}},
    {{1, 37, 9}}
};

INSTANTIATE_TEST_SUITE_P(smoke_Decomposition_3D, Mvn6LayerTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shapes_decomposition_3d_static)),
                            ::testing::ValuesIn(model_types),
                            ::testing::ValuesIn(idx_types),
                            ::testing::ValuesIn(std::vector<std::vector<int>>{{0, 1, 2}, {0}, {1}}),
                            ::testing::ValuesIn(normalize_variance),
                            ::testing::ValuesIn(epsilonF),
                            ::testing::ValuesIn(epsMode),
                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        Mvn6LayerTest::getTestCaseName);

const std::vector<std::vector<ov::Shape>> input_shapes_decomposition_4d_static = {
    {{1, 16, 5, 8}},
    {{2, 19, 5, 10}}
};

INSTANTIATE_TEST_SUITE_P(smoke_Decomposition_4D, Mvn6LayerTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shapes_decomposition_4d_static)),
                            ::testing::ValuesIn(model_types),
                            ::testing::ValuesIn(idx_types),
                            ::testing::ValuesIn(std::vector<std::vector<int>>{{0, 1, 2, 3}, {0, 1, 2}, {0, 3}, {0}, {1}, {2}, {3}}),
                            ::testing::ValuesIn(normalize_variance),
                            ::testing::ValuesIn(epsilonF),
                            ::testing::ValuesIn(epsMode),
                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        Mvn6LayerTest::getTestCaseName);

const std::vector<std::vector<ov::Shape>> input_shapes_decomposition_10d_static = {
    {{1, 3, 5, 4, 2, 6, 5, 3, 2, 1}},
};

INSTANTIATE_TEST_SUITE_P(smoke_Decomposition_10D, Mvn6LayerTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shapes_decomposition_10d_static)),
                            ::testing::ValuesIn(model_types),
                            ::testing::ValuesIn(idx_types),
                            ::testing::ValuesIn(std::vector<std::vector<int>>{{0, 1, 5, 8, 9}, {0, 1, 2, 3}, {0, 1, 2}, {0, 3}, {0}, {3}, {5}, {9}}),
                            ::testing::ValuesIn(normalize_variance),
                            ::testing::ValuesIn(epsilonF),
                            ::testing::ValuesIn(epsMode),
                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        Mvn6LayerTest::getTestCaseName);
} // namespace