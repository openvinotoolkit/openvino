// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "subgraph_tests/mul_conv_fusion.hpp"
#include "common_test_utils/test_constants.hpp"

#include <ngraph/opsets/opset8.hpp>

using namespace SubgraphTestsDefinitions;

namespace {
    const std::vector<ngraph::element::Type> types{ngraph::element::f32, ngraph::element::f16};

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
                                    ::testing::Values(ngraph::opset8::Convolution::type_info),
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
        {3, 1, 1},
        {1, 1, 1, 1},
        {1, 3, 1, 1},
    };

    INSTANTIATE_TEST_SUITE_P(smoke_Convolution_2D, MulConvFusion,
                            ::testing::Combine(
                                    ::testing::Values(ngraph::opset8::Convolution::type_info),
                                    ::testing::Values(ngraph::Shape{1, 3, 64, 64}),
                                    ::testing::Values(ngraph::Shape{20, 3, 4, 4}),
                                    ::testing::ValuesIn(const_shapes_fprop_2d),
                                    ::testing::ValuesIn(types),
                                    ::testing::Values(false), // Positive test
                                    ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                            MulConvFusion::getTestCaseName);

    const std::vector<ngraph::Shape> const_shapes_others{
        {},
        {1},
        {1, 1},
        {1, 1, 1},
        {1, 1, 1, 1},
    };

    INSTANTIATE_TEST_SUITE_P(smoke_ConvolutionBackpropData_2D, MulConvFusion,
                            ::testing::Combine(
                                    ::testing::Values(ngraph::opset8::ConvolutionBackpropData::type_info),
                                    ::testing::Values(ngraph::Shape{1, 3, 64, 64}),
                                    ::testing::Values(ngraph::Shape{3, 20, 3, 3}),
                                    ::testing::ValuesIn(const_shapes_others),
                                    ::testing::ValuesIn(types),
                                    ::testing::Values(false), // Positive test
                                    ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                            MulConvFusion::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_GroupConvolution_2D, MulConvFusion,
                            ::testing::Combine(
                                    ::testing::Values(ngraph::opset8::GroupConvolution::type_info),
                                    ::testing::Values(ngraph::Shape{1, 12, 64, 64}),
                                    ::testing::Values(ngraph::Shape{4, 5, 3, 1, 2}),
                                    ::testing::ValuesIn(const_shapes_others),
                                    ::testing::ValuesIn(types),
                                    ::testing::Values(false), // Positive test
                                    ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                            MulConvFusion::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_GroupConvolutionBackpropData_2D, MulConvFusion,
                            ::testing::Combine(
                                    ::testing::Values(ngraph::opset8::GroupConvolutionBackpropData::type_info),
                                    ::testing::Values(ngraph::Shape{1, 12, 3, 3}),
                                    ::testing::Values(ngraph::Shape{4, 3, 5, 1, 1}),
                                    ::testing::ValuesIn(const_shapes_others),
                                    ::testing::ValuesIn(types),
                                    ::testing::Values(false), // Positive test
                                    ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                            MulConvFusion::getTestCaseName);

    const std::vector<ngraph::Shape> negative_const_shapes_fprop_2d{
        {3, 64, 64},
        {1, 3, 64, 64},
    };

    INSTANTIATE_TEST_SUITE_P(smoke_NegativeConvolution_2D, MulConvFusion,
                            ::testing::Combine(
                                    ::testing::Values(ngraph::opset8::Convolution::type_info),
                                    ::testing::Values(ngraph::Shape{1, 3, 64, 64}),
                                    ::testing::Values(ngraph::Shape{20, 3, 1, 1}),
                                    ::testing::ValuesIn(negative_const_shapes_fprop_2d),
                                    ::testing::ValuesIn(types),
                                    ::testing::Values(true), // Negative test
                                    ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                            MulConvFusion::getTestCaseName);

    const std::vector<ngraph::Shape> negative_const_shapes_others{
        {12, 1, 1},
        {1, 12, 1, 1},
        {1, 1, 64, 64},
    };

    INSTANTIATE_TEST_SUITE_P(smoke_NegativeConvolutionBackpropData_2D, MulConvFusion,
                            ::testing::Combine(
                                    ::testing::Values(ngraph::opset8::ConvolutionBackpropData::type_info),
                                    ::testing::Values(ngraph::Shape{1, 12, 64, 64}),
                                    ::testing::Values(ngraph::Shape{12, 20, 3, 3}),
                                    ::testing::ValuesIn(negative_const_shapes_others),
                                    ::testing::ValuesIn(types),
                                    ::testing::Values(true), // Negative test
                                    ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                            MulConvFusion::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_NegativeGroupConvolution_2D, MulConvFusion,
                            ::testing::Combine(
                                    ::testing::Values(ngraph::opset8::GroupConvolution::type_info),
                                    ::testing::Values(ngraph::Shape{1, 12, 64, 64}),
                                    ::testing::Values(ngraph::Shape{4, 5, 3, 1, 2}),
                                    ::testing::ValuesIn(negative_const_shapes_others),
                                    ::testing::ValuesIn(types),
                                    ::testing::Values(true), // Negative test
                                    ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                            MulConvFusion::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_NegativeGroupConvolutionBackpropData_2D, MulConvFusion,
                            ::testing::Combine(
                                    ::testing::Values(ngraph::opset8::GroupConvolutionBackpropData::type_info),
                                    ::testing::Values(ngraph::Shape{1, 12, 64, 64}),
                                    ::testing::Values(ngraph::Shape{4, 3, 5, 1, 1}),
                                    ::testing::ValuesIn(negative_const_shapes_others),
                                    ::testing::ValuesIn(types),
                                    ::testing::Values(true), // Negative test
                                    ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                            MulConvFusion::getTestCaseName);

}  // namespace
