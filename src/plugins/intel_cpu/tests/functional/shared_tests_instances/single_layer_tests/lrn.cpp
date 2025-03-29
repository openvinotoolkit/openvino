// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/lrn.hpp"
#include "common_test_utils/test_constants.hpp"

namespace {
using ov::test::LrnLayerTest;

// common values

const std::vector<ov::element::Type> model_types{ov::element::f32};
const double alpha = 9.9e-05;
const double beta = 2;
const double bias = 1.0;
const size_t size = 5;

// 2D

const std::vector<std::vector<int64_t>> axes_2d = {{1}};
std::vector<ov::Shape> input_shapes_2d_static = {{10, 16}};

INSTANTIATE_TEST_SUITE_P(smoke_LrnCheck_2D, LrnLayerTest,
                        ::testing::Combine(::testing::Values(alpha),
                                           ::testing::Values(beta),
                                           ::testing::Values(bias),
                                           ::testing::Values(size),
                                           ::testing::ValuesIn(axes_2d),
                                           ::testing::ValuesIn(model_types),
                                           ::testing::Values(ov::test::static_shapes_to_test_representation(input_shapes_2d_static)),
                                           ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        LrnLayerTest::getTestCaseName);


// 3D

const std::vector<std::vector<int64_t>> axes_3d = {{1}, {2}};
std::vector<ov::Shape> input_shapes_3d_static = {{6, 10, 16}};

INSTANTIATE_TEST_SUITE_P(smoke_LrnCheck_3D, LrnLayerTest,
                        ::testing::Combine(::testing::Values(alpha),
                                           ::testing::Values(beta),
                                           ::testing::Values(bias),
                                           ::testing::Values(size),
                                           ::testing::ValuesIn(axes_3d),
                                           ::testing::ValuesIn(model_types),
                                           ::testing::Values(ov::test::static_shapes_to_test_representation(input_shapes_3d_static)),
                                           ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        LrnLayerTest::getTestCaseName);



// 4D

const std::vector<std::vector<int64_t>> axes_4d = {{1}, {2, 3}, {3, 2}};
std::vector<ov::Shape> input_shapes_4d_static = {{10, 10, 3, 8}};

INSTANTIATE_TEST_SUITE_P(smoke_LrnCheck_4D, LrnLayerTest,
                        ::testing::Combine(::testing::Values(alpha),
                                           ::testing::Values(beta),
                                           ::testing::Values(bias),
                                           ::testing::Values(size),
                                           ::testing::ValuesIn(axes_4d),
                                           ::testing::ValuesIn(model_types),
                                           ::testing::Values(ov::test::static_shapes_to_test_representation(input_shapes_4d_static)),
                                           ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        LrnLayerTest::getTestCaseName);



// 5D

const std::vector<std::vector<int64_t>> axes_5d = {{1}, {2, 3, 4}, {4, 2, 3}};
std::vector<ov::Shape> input_shapes_5d_static = {{1, 10, 10, 7, 4}};


INSTANTIATE_TEST_SUITE_P(smoke_LrnCheck_5D, LrnLayerTest,
                        ::testing::Combine(::testing::Values(alpha),
                                           ::testing::Values(beta),
                                           ::testing::Values(bias),
                                           ::testing::Values(size),
                                           ::testing::ValuesIn(axes_5d),
                                           ::testing::ValuesIn(model_types),
                                           ::testing::Values(ov::test::static_shapes_to_test_representation(input_shapes_5d_static)),
                                           ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        LrnLayerTest::getTestCaseName);

} // namespace
