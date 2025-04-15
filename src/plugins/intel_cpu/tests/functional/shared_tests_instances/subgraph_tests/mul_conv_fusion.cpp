// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/mul_conv_fusion.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"

using namespace ov::test;

namespace {
const std::vector<ov::element::Type> types{ov::element::f32};

const std::vector<ov::Shape> const_shapes_fprop_1d{
    {},
    {1},
    {1, 1},
    {8, 1},
    {1, 1, 1},
    {1, 8, 1},
};

INSTANTIATE_TEST_SUITE_P(smoke_Convolution_1D,
                         MulConvFusion,
                         ::testing::Combine(::testing::Values(ov::op::v1::Convolution::get_type_info_static()),
                                            ::testing::Values(ov::Shape{1, 8, 64}),
                                            ::testing::Values(ov::Shape{64, 8, 1}),
                                            ::testing::ValuesIn(const_shapes_fprop_1d),
                                            ::testing::ValuesIn(types),
                                            ::testing::Values(false),  // Positive test
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         MulConvFusion::getTestCaseName);

const std::vector<ov::Shape> const_shapes_fprop_2d{
    {},
    {1},
    {1, 1},
    {1, 1, 1},
    {8, 1, 1},
    {1, 1, 1, 1},
    {1, 8, 1, 1},
};

INSTANTIATE_TEST_SUITE_P(smoke_Convolution_2D,
                         MulConvFusion,
                         ::testing::Combine(::testing::Values(ov::op::v1::Convolution::get_type_info_static()),
                                            ::testing::Values(ov::Shape{2, 8, 14, 14}),
                                            ::testing::Values(ov::Shape{2, 8, 7, 7}),
                                            ::testing::ValuesIn(const_shapes_fprop_2d),
                                            ::testing::ValuesIn(types),
                                            ::testing::Values(false),  // Positive test
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         MulConvFusion::getTestCaseName);

const std::vector<ov::Shape> const_shapes_fprop_2d_kernel_same_as_input{
    {7},
    {1, 7},
    {1, 1, 7},
    {8, 1, 7},
    {1, 1, 1, 7},
    {1, 8, 1, 7},
    {7, 1},
    {1, 7, 1},
    {8, 7, 1},
    {1, 1, 7, 1},
    {1, 8, 7, 1},
    {1, 1, 7, 7},
    {1, 8, 7, 7},
};

INSTANTIATE_TEST_SUITE_P(smoke_Convolution_2D_kernel_same_as_input,
                         MulConvFusion,
                         ::testing::Combine(::testing::Values(ov::op::v1::Convolution::get_type_info_static()),
                                            ::testing::Values(ov::Shape{2, 8, 7, 7}),
                                            ::testing::Values(ov::Shape{3, 8, 7, 7}),
                                            ::testing::ValuesIn(const_shapes_fprop_2d_kernel_same_as_input),
                                            ::testing::ValuesIn(types),
                                            ::testing::Values(false),  // Positive test
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         MulConvFusion::getTestCaseName);

const std::vector<ov::Shape> const_shapes_conv_bprop{
    {},
    {1},
    {1, 1},
    {1, 1, 1},
    {1, 1, 1, 1},
    {3, 1, 1},
    {1, 3, 1, 1},
};
INSTANTIATE_TEST_SUITE_P(
    smoke_ConvolutionBackpropData_2D,
    MulConvFusion,
    ::testing::Combine(::testing::Values(ov::op::v1::ConvolutionBackpropData::get_type_info_static()),
                       ::testing::Values(ov::Shape{4, 3, 64, 64}),
                       ::testing::Values(ov::Shape{3, 20, 3, 3}),
                       ::testing::ValuesIn(const_shapes_conv_bprop),
                       ::testing::ValuesIn(types),
                       ::testing::Values(false),  // Positive test
                       ::testing::Values(ov::test::utils::DEVICE_CPU)),
    MulConvFusion::getTestCaseName);

const std::vector<ov::Shape> const_shapes_group_conv{
    {},
    {1},
    {1, 1},
    {1, 1, 1},
    {12, 1, 1},
    {1, 1, 1, 1},
    {1, 12, 1, 1},
};

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvolution_2D,
                         MulConvFusion,
                         ::testing::Combine(::testing::Values(ov::op::v1::GroupConvolution::get_type_info_static()),
                                            ::testing::Values(ov::Shape{2, 12, 14, 14}),
                                            ::testing::Values(ov::Shape{4, 5, 3, 7, 7}),
                                            ::testing::ValuesIn(const_shapes_group_conv),
                                            ::testing::ValuesIn(types),
                                            ::testing::Values(false),  // Positive test
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         MulConvFusion::getTestCaseName);

const std::vector<ov::Shape> const_shapes_group_conv_kernel_same_as_input{
    {14},
    {1, 14},
    {1, 1, 14},
    {12, 1, 14},
    {1, 1, 1, 14},
    {1, 12, 1, 14},
    {14, 1},
    {1, 14, 1},
    {12, 14, 1},
    {1, 1, 14, 1},
    {1, 12, 14, 1},
    {1, 1, 14, 14},
    {1, 12, 14, 14},
};

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvolution_2D_kernel_same_as_input,
                         MulConvFusion,
                         ::testing::Combine(::testing::Values(ov::op::v1::GroupConvolution::get_type_info_static()),
                                            ::testing::Values(ov::Shape{2, 12, 14, 14}),
                                            ::testing::Values(ov::Shape{4, 5, 3, 14, 14}),
                                            ::testing::ValuesIn(const_shapes_group_conv_kernel_same_as_input),
                                            ::testing::ValuesIn(types),
                                            ::testing::Values(false),  // Positive test
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         MulConvFusion::getTestCaseName);

const std::vector<ov::Shape> const_shapes_group_conv_bprop{
    {},
    {1},
    {1, 1},
    {1, 1, 1},
    {12, 1, 1},
    {1, 1, 1, 1},
    {1, 12, 1, 1},
};

INSTANTIATE_TEST_SUITE_P(
    smoke_GroupConvolutionBackpropData_2D,
    MulConvFusion,
    ::testing::Combine(::testing::Values(ov::op::v1::GroupConvolutionBackpropData::get_type_info_static()),
                       ::testing::Values(ov::Shape{2, 12, 10, 10}),
                       ::testing::Values(ov::Shape{4, 3, 5, 2, 2}),
                       ::testing::ValuesIn(const_shapes_group_conv_bprop),
                       ::testing::ValuesIn(types),
                       ::testing::Values(false),  // Positive test
                       ::testing::Values(ov::test::utils::DEVICE_CPU)),
    MulConvFusion::getTestCaseName);

const std::vector<ov::Shape> negative_const_shapes{
    {12, 64, 64},
    {2, 1, 1, 1},
    {1, 1, 64, 64},
    {1, 12, 64, 64},
    {2, 12, 64, 64},
};

INSTANTIATE_TEST_SUITE_P(smoke_NegativeConvolution_2D,
                         MulConvFusion,
                         ::testing::Combine(::testing::Values(ov::op::v1::Convolution::get_type_info_static()),
                                            ::testing::Values(ov::Shape{2, 12, 64, 64}),
                                            ::testing::Values(ov::Shape{20, 12, 1, 1}),
                                            ::testing::ValuesIn(negative_const_shapes),
                                            ::testing::ValuesIn(types),
                                            ::testing::Values(true),  // Negative test
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         MulConvFusion::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_NegativeConvolutionBackpropData_2D,
    MulConvFusion,
    ::testing::Combine(::testing::Values(ov::op::v1::ConvolutionBackpropData::get_type_info_static()),
                       ::testing::Values(ov::Shape{2, 12, 64, 64}),
                       ::testing::Values(ov::Shape{12, 20, 3, 3}),
                       ::testing::ValuesIn(negative_const_shapes),
                       ::testing::ValuesIn(types),
                       ::testing::Values(true),  // Negative test
                       ::testing::Values(ov::test::utils::DEVICE_CPU)),
    MulConvFusion::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_NegativeGroupConvolution_2D,
                         MulConvFusion,
                         ::testing::Combine(::testing::Values(ov::op::v1::GroupConvolution::get_type_info_static()),
                                            ::testing::Values(ov::Shape{2, 12, 64, 64}),
                                            ::testing::Values(ov::Shape{4, 5, 3, 1, 2}),
                                            ::testing::ValuesIn(negative_const_shapes),
                                            ::testing::ValuesIn(types),
                                            ::testing::Values(true),  // Negative test
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         MulConvFusion::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_NegativeGroupConvolutionBackpropData_2D,
    MulConvFusion,
    ::testing::Combine(::testing::Values(ov::op::v1::GroupConvolutionBackpropData::get_type_info_static()),
                       ::testing::Values(ov::Shape{2, 12, 64, 64}),
                       ::testing::Values(ov::Shape{4, 3, 5, 1, 1}),
                       ::testing::ValuesIn(negative_const_shapes),
                       ::testing::ValuesIn(types),
                       ::testing::Values(true),  // Negative test
                       ::testing::Values(ov::test::utils::DEVICE_CPU)),
    MulConvFusion::getTestCaseName);

const std::vector<ov::Shape> negative_const_shapes_kernel_same_as_input{
    {7},
    {1, 7},
    {1, 1, 7},
    {12, 1, 7},
    {1, 1, 1, 7},
    {1, 12, 1, 7},
    {7, 1},
    {1, 7, 1},
    {12, 7, 1},
    {1, 1, 7, 1},
    {1, 12, 7, 1},
    {1, 1, 7, 7},
    {1, 12, 7, 7},
};

INSTANTIATE_TEST_SUITE_P(
    smoke_NegativeConvolutionBackpropData_2D_kernel_same_as_input,
    MulConvFusion,
    ::testing::Combine(::testing::Values(ov::op::v1::ConvolutionBackpropData::get_type_info_static()),
                       ::testing::Values(ov::Shape{2, 12, 7, 7}),
                       ::testing::Values(ov::Shape{12, 20, 7, 7}),
                       ::testing::ValuesIn(negative_const_shapes_kernel_same_as_input),
                       ::testing::ValuesIn(types),
                       ::testing::Values(true),  // Negative test
                       ::testing::Values(ov::test::utils::DEVICE_CPU)),
    MulConvFusion::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_NegativeGroupConvolutionBackpropData_2D_kernel_same_as_input,
    MulConvFusion,
    ::testing::Combine(::testing::Values(ov::op::v1::GroupConvolutionBackpropData::get_type_info_static()),
                       ::testing::Values(ov::Shape{2, 12, 7, 7}),
                       ::testing::Values(ov::Shape{4, 3, 5, 7, 7}),
                       ::testing::ValuesIn(negative_const_shapes_kernel_same_as_input),
                       ::testing::ValuesIn(types),
                       ::testing::Values(true),  // Negative test
                       ::testing::Values(ov::test::utils::DEVICE_CPU)),
    MulConvFusion::getTestCaseName);

}  // namespace
