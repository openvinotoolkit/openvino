// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "subgraph_tests/mul_conv_fusion.hpp"
#include "common_test_utils/test_constants.hpp"

#include <ngraph/opsets/opset8.hpp>

using namespace SubgraphTestsDefinitions;

namespace {
    const std::vector<ngraph::element::Type> types{ngraph::element::f32};

    const std::vector<ngraph::Shape> const_shapes_fprop_1d{
        {},
        {1},
        {1, 1},
        {8, 1},
        {1, 1, 1},
        {1, 8, 1},
    };

    INSTANTIATE_TEST_SUITE_P(smoke_Convolution_1D, MulConvFusion,
                            ::testing::Combine(
                                    ::testing::Values(ngraph::opset8::Convolution::get_type_info_static()),
                                    ::testing::Values(ngraph::Shape{1, 8, 64}),
                                    ::testing::Values(ngraph::Shape{64, 8, 1}),
                                    ::testing::ValuesIn(const_shapes_fprop_1d),
                                    ::testing::ValuesIn(types),
                                    ::testing::Values(false), // Positive test
                                    ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                            MulConvFusion::getTestCaseName);

    const std::vector<ngraph::Shape> const_shapes_fprop_2d{
        {},
        {1},
        {1, 1},
        {1, 1, 1},
        {8, 1, 1},
        {1, 1, 1, 1},
        {1, 8, 1, 1},
    };

    INSTANTIATE_TEST_SUITE_P(smoke_Convolution_2D, MulConvFusion,
                            ::testing::Combine(
                                    ::testing::Values(ngraph::opset8::Convolution::get_type_info_static()),
                                    ::testing::Values(ngraph::Shape{2, 8, 14, 14}),
                                    ::testing::Values(ngraph::Shape{2, 8, 7, 7}),
                                    ::testing::ValuesIn(const_shapes_fprop_2d),
                                    ::testing::ValuesIn(types),
                                    ::testing::Values(false), // Positive test
                                    ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                            MulConvFusion::getTestCaseName);

    const std::vector<ngraph::Shape> const_shapes_fprop_2d_kernel_same_as_input{
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

    INSTANTIATE_TEST_SUITE_P(smoke_Convolution_2D_kernel_same_as_input, MulConvFusion,
                            ::testing::Combine(
                                    ::testing::Values(ngraph::opset8::Convolution::get_type_info_static()),
                                    ::testing::Values(ngraph::Shape{2, 8, 7, 7}),
                                    ::testing::Values(ngraph::Shape{3, 8, 7, 7}),
                                    ::testing::ValuesIn(const_shapes_fprop_2d_kernel_same_as_input),
                                    ::testing::ValuesIn(types),
                                    ::testing::Values(false), // Positive test
                                    ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                            MulConvFusion::getTestCaseName);

    const std::vector<ngraph::Shape> const_shapes_conv_bprop{
        {},
        {1},
        {1, 1},
        {1, 1, 1},
        {1, 1, 1, 1},
        {3, 1, 1},
        {1, 3, 1, 1},
    };
    INSTANTIATE_TEST_SUITE_P(smoke_ConvolutionBackpropData_2D, MulConvFusion,
                            ::testing::Combine(
                                    ::testing::Values(ngraph::opset8::ConvolutionBackpropData::get_type_info_static()),
                                    ::testing::Values(ngraph::Shape{4, 3, 64, 64}),
                                    ::testing::Values(ngraph::Shape{3, 20, 3, 3}),
                                    ::testing::ValuesIn(const_shapes_conv_bprop),
                                    ::testing::ValuesIn(types),
                                    ::testing::Values(false), // Positive test
                                    ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                            MulConvFusion::getTestCaseName);

    const std::vector<ngraph::Shape> const_shapes_group_conv{
        {},
        {1},
        {1, 1},
        {1, 1, 1},
        {12, 1, 1},
        {1, 1, 1, 1},
        {1, 12, 1, 1},
    };

    INSTANTIATE_TEST_SUITE_P(smoke_GroupConvolution_2D, MulConvFusion,
                            ::testing::Combine(
                                    ::testing::Values(ngraph::opset8::GroupConvolution::get_type_info_static()),
                                    ::testing::Values(ngraph::Shape{2, 12, 14, 14}),
                                    ::testing::Values(ngraph::Shape{4, 5, 3, 7, 7}),
                                    ::testing::ValuesIn(const_shapes_group_conv),
                                    ::testing::ValuesIn(types),
                                    ::testing::Values(false), // Positive test
                                    ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                            MulConvFusion::getTestCaseName);

    const std::vector<ngraph::Shape> const_shapes_group_conv_kernel_same_as_input{
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

    INSTANTIATE_TEST_SUITE_P(smoke_GroupConvolution_2D_kernel_same_as_input, MulConvFusion,
                            ::testing::Combine(
                                    ::testing::Values(ngraph::opset8::GroupConvolution::get_type_info_static()),
                                    ::testing::Values(ngraph::Shape{2, 12, 14, 14}),
                                    ::testing::Values(ngraph::Shape{4, 5, 3, 14, 14}),
                                    ::testing::ValuesIn(const_shapes_group_conv_kernel_same_as_input),
                                    ::testing::ValuesIn(types),
                                    ::testing::Values(false), // Positive test
                                    ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                            MulConvFusion::getTestCaseName);

    const std::vector<ngraph::Shape> const_shapes_group_conv_bprop{
        {},
        {1},
        {1, 1},
        {1, 1, 1},
        {12, 1, 1},
        {1, 1, 1, 1},
        {1, 12, 1, 1},
    };

    INSTANTIATE_TEST_SUITE_P(smoke_GroupConvolutionBackpropData_2D, MulConvFusion,
                            ::testing::Combine(
                                    ::testing::Values(ngraph::opset8::GroupConvolutionBackpropData::get_type_info_static()),
                                    ::testing::Values(ngraph::Shape{2, 12, 10, 10}),
                                    ::testing::Values(ngraph::Shape{4, 3, 5, 2, 2}),
                                    ::testing::ValuesIn(const_shapes_group_conv_bprop),
                                    ::testing::ValuesIn(types),
                                    ::testing::Values(false), // Positive test
                                    ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                            MulConvFusion::getTestCaseName);

    const std::vector<ngraph::Shape> negative_const_shapes{
        {12, 64, 64},
        {2, 1, 1, 1},
        {1, 1, 64, 64},
        {1, 12, 64, 64},
        {2, 12, 64, 64},
    };

    INSTANTIATE_TEST_SUITE_P(smoke_NegativeConvolution_2D, MulConvFusion,
                            ::testing::Combine(
                                    ::testing::Values(ngraph::opset8::Convolution::get_type_info_static()),
                                    ::testing::Values(ngraph::Shape{2, 12, 64, 64}),
                                    ::testing::Values(ngraph::Shape{20, 12, 1, 1}),
                                    ::testing::ValuesIn(negative_const_shapes),
                                    ::testing::ValuesIn(types),
                                    ::testing::Values(true), // Negative test
                                    ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                            MulConvFusion::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_NegativeConvolutionBackpropData_2D, MulConvFusion,
                            ::testing::Combine(
                                    ::testing::Values(ngraph::opset8::ConvolutionBackpropData::get_type_info_static()),
                                    ::testing::Values(ngraph::Shape{2, 12, 64, 64}),
                                    ::testing::Values(ngraph::Shape{12, 20, 3, 3}),
                                    ::testing::ValuesIn(negative_const_shapes),
                                    ::testing::ValuesIn(types),
                                    ::testing::Values(true), // Negative test
                                    ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                            MulConvFusion::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_NegativeGroupConvolution_2D, MulConvFusion,
                            ::testing::Combine(
                                    ::testing::Values(ngraph::opset8::GroupConvolution::get_type_info_static()),
                                    ::testing::Values(ngraph::Shape{2, 12, 64, 64}),
                                    ::testing::Values(ngraph::Shape{4, 5, 3, 1, 2}),
                                    ::testing::ValuesIn(negative_const_shapes),
                                    ::testing::ValuesIn(types),
                                    ::testing::Values(true), // Negative test
                                    ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                            MulConvFusion::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_NegativeGroupConvolutionBackpropData_2D, MulConvFusion,
                            ::testing::Combine(
                                    ::testing::Values(ngraph::opset8::GroupConvolutionBackpropData::get_type_info_static()),
                                    ::testing::Values(ngraph::Shape{2, 12, 64, 64}),
                                    ::testing::Values(ngraph::Shape{4, 3, 5, 1, 1}),
                                    ::testing::ValuesIn(negative_const_shapes),
                                    ::testing::ValuesIn(types),
                                    ::testing::Values(true), // Negative test
                                    ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                            MulConvFusion::getTestCaseName);

    const std::vector<ngraph::Shape> negative_const_shapes_kernel_same_as_input{
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

    INSTANTIATE_TEST_SUITE_P(smoke_NegativeConvolutionBackpropData_2D_kernel_same_as_input, MulConvFusion,
                            ::testing::Combine(
                                    ::testing::Values(ngraph::opset8::ConvolutionBackpropData::get_type_info_static()),
                                    ::testing::Values(ngraph::Shape{2, 12, 7, 7}),
                                    ::testing::Values(ngraph::Shape{12, 20, 7, 7}),
                                    ::testing::ValuesIn(negative_const_shapes_kernel_same_as_input),
                                    ::testing::ValuesIn(types),
                                    ::testing::Values(true), // Negative test
                                    ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                            MulConvFusion::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_NegativeGroupConvolutionBackpropData_2D_kernel_same_as_input, MulConvFusion,
                            ::testing::Combine(
                                    ::testing::Values(ngraph::opset8::GroupConvolutionBackpropData::get_type_info_static()),
                                    ::testing::Values(ngraph::Shape{2, 12, 7, 7}),
                                    ::testing::Values(ngraph::Shape{4, 3, 5, 7, 7}),
                                    ::testing::ValuesIn(negative_const_shapes_kernel_same_as_input),
                                    ::testing::ValuesIn(types),
                                    ::testing::Values(true), // Negative test
                                    ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                            MulConvFusion::getTestCaseName);

}  // namespace
